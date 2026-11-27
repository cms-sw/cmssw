# MkFinderV2p2 design notes

## What v2p2 is

mkFit finds tracks by following each seed through an ordered list of layers. In
each layer it propagates every candidate track to the layer, looks for
compatible hits, updates the candidate with a Kalman filter, and keeps the best
few candidates of the seed. There are three implementations of that layer step:

- **V1**, the original: a search window from the propagated state, one Kalman
  update per hit from the incoming state, and at most two hits per layer, the
  second from an overlapping module.
- **V2**, the same with a newer hit selection (`MkFinder::selectHitIndicesV2()`).
  It is what CMSSW runs for mkFit in production.
- **v2p2**, this one, a rewrite of the whole layer step: a window built from the
  track's covariance at the layer, the crossing with each hit's own module plane,
  an in-layer search that can take several hits per layer (both sensors of a
  module, overlapping modules), and one selection per seed on a score that adds
  up layer by layer. In CMSSW it runs for the phase-2 initialStep
  (`buildingRoutine = 'v2p2'`).

The overview at the top of `MkFinderV2p2.cc` says what the code does. These notes
say why. Most sections start with a **Code:** line naming the classes and
functions they are about; the text uses as few other names as it can.

## Words used here

| word | meaning |
|---|---|
| seed | the short track a search starts from |
| CombCandidate | everything grown from one seed: up to `maxCandsPerSeed` candidates, the seed's beam |
| TrackCand, candidate | one candidate track of a CombCandidate |
| beam width | `maxCandsPerSeed`, the number of candidates a seed keeps after each layer |
| layer plan | the ordered list of layers searched, one for each of five eta regions: barrel, and transition and endcap on each side (see below) |
| pickup | the plan entry holding the layer of a seed's last hit; the seed is searched from the next entry on |
| HoT, hole | hit on track: one entry per hit taken in a layer (v2p2 can take several), or a hole. A hole is a HoT with no hit, of one of four kinds: miss (counts against the hole limits), stop (the candidate ends there), edge (the track may have passed beside the modules) and gap (it passed through an opening of a disc); edge and gap are not counted. Once a track is complete, a hole is an inside hole if a hit follows it and a tail hole otherwise |
| WSR | within sensitive region: whether the track crosses the layer inside, at the edge of, or outside its modules |
| q, qbar, phi | the coordinates of a layer: q is z in the barrel and r in the endcap, qbar is the other one (r in the barrel, z in the endcap), phi the azimuth |
| binnor | the per-layer (phi, q) grid that hits are sorted into, so that a window is a range of bins |
| window, fetch, cut | the bins read for a candidate (fetch) and the per-hit test applied to what they hold (cut) |
| overlap | a second module of the same layer crossed by the track where neighbouring modules overlap |
| dalpha | the turning angle of the track from its state to a point; it orders hits along the path |
| path | the hits one candidate takes in one layer, ordered as the track meets them |
| layer step | what one candidate does in one layer; it is scored as one additive term |
| forward, inward search | outward, from pixel seeds; inward, towards the beam, as from outer-tracker seeds. For regions in \|eta\| these notes say barrel, transition and endcap |
| mini-propagators | mkFit's light propagation of track parameters only, without the covariance |
| Matriplex, NN | mkFit's SIMD matrix batches, NN tracks wide: 8 lanes with AVX, 16 with AVX-512 |
| initialStep | the first iteration of CMS tracking, seeded from pixel tracks |
| LST, T5 | Line Segment Tracking, the phase-2 outer-tracker pattern recognition; a T5 is its track of five outer-tracker modules |
| quality-val, MTV | mkFit's own efficiency validation, and CMSSW's MultiTrackValidator |

**Layers and the plan.** Phase-2 names: the OT PS barrel is OT layers 1-3, the
OT 2S barrel is OT layers 4-6, and the OT endcap has PS modules at small radius
and 2S at large. P and S are the two sensors of a PS module: P macro-pixels
(1.5 mm long), S strips (24 mm). mkFit gives each sensor layer of the outer
tracker its own layer number: OT layers 1-3 are mkFit layers 4-9, OT layers 4-6
are 10-15, and each endcap disc is two mkFit layers. In the default plan, the
**split plan**, each mkFit layer is its own plan entry: the two sensors of a
module are searched one after the other, each from the state after the
previous entry, and a missed sensor records its own hole. A **paired** entry,
one search over both sensors with a shared window, is supported but not used by
default. Within one mkFit layer a track still crosses two or more modules where
modules overlap.

Contents:

1. [Samples and how numbers are quoted](#samples-and-how-numbers-are-quoted)
2. [Configuration](#configuration)
3. [Layer processing](#layer-processing)
4. [Candidate pickup and stopping cuts](#candidate-pickup-and-stopping-cuts)
5. [Layer crossings and the Hermite cubic](#layer-crossings-and-the-hermite-cubic)
6. [Reachability](#reachability)
7. [Within-sensitive-region verdict](#within-sensitive-region-verdict)
8. [Hit extents](#hit-extents)
9. [Search window](#search-window)
10. [Line pre-cut](#line-pre-cut)
11. [Reduction and hit ordering](#reduction-and-hit-ordering)
12. [In-layer combinatorial search](#in-layer-combinatorial-search)
13. [Kalman hit acceptance](#kalman-hit-acceptance)
14. [Grazing incidence](#grazing-incidence)
15. [End-of-layer selection](#end-of-layer-selection)
16. [Layer-step score](#layer-step-score)
17. [Backward fit](#backward-fit)
18. [V2 compatibility](#v2-compatibility)
19. [Open items](#open-items)
20. [Transport of the position
    covariance](#appendix-transport-of-the-position-covariance)
21. [History](#appendix-history)

## Samples and how numbers are quoted

All measurements use ttbar events with pileup 200 in the phase-2 tracker
geometry. The forward search runs from the CMSSW initialStep seeds, the inward
search from T5s from LST and has to add their pixel hits. The samples used
repeatedly are named here and by tag afterwards.

- **F30**: forward search, 30 events. Efficiency with quality-val's matching,
  against an MTV-like denominator (29759 selected sim tracks).
- **I50**: inward search, 50 events. Counted: pixel hits recovered, out of the
  211435 pixel hits LST had found for these tracks in the HLT setup (the search
  starts from the same tracks without their pixel hits), matched by
  exact (layer, index); and fully recovered tracks, those with all of them, out
  of 43358. Neither count uses truth matching.
- **T100**: time of the forward search alone (not seeding, fitting or duplicate
  removal), 100 events, single thread, minimum of three repetitions. The
  machine is named where it is known.

How to read the numbers:

- **Comparisons are paired**: same events, same seeds, one process, one
  setting changed. A difference like "-41 found" is the changed setting minus
  the reference, which the text names. A significance in parentheses is that
  difference over its event-to-event spread (sqrt(n) times the standard
  deviation of the per-event differences).
- **quality-val's found and fake counts reward track length.** In its default
  mode a track matched to a sim track but with fewer than 10 found hits (seed
  hits included) is not counted as found; it is counted with the tracks that
  match no sim track, which the tables call fakes. So is a track on a particle
  from which no seed was built. A change that makes tracks longer therefore gains
  found tracks and loses fakes without being purer. Where it matters, MTV's
  association (a fake is a track matched to no sim track, no minimum length) is
  quoted as well. **Found and fake counts on F30 use quality-val's default rule
  unless MTV is named.**
- **Times are of the code as it was when measured.** Compare times within one
  table only: the absolute T100 times of different sections were taken at
  different states of the code. Older ones are in [History](#appendix-history).

## Configuration

v2p2 reads three kinds of parameters.

| | what | where | set by |
|---|---|---|---|
| iteration parameters | beam width, minimum pT, hole limits, the track scorers per task | `IterationConfig`, as for V1 and V2 | the iteration's JSON configuration |
| v2p2 tunings | the two score sets, the hit-acceptance cut, the hit caps (`max_presel_hits`, `max_hits_per_layer`) | `interface/V2p2Params.h`, `IterationConfig::m_v2p2` | the python parameter set `v2p2` of `MkFitIterationConfigESProducer` |
| v2p2 knobs | search window, pre-cut, in-layer search, stopping policy, score form | `src/V2p2Config.h`, namespace `Config::V2p2` | the code: constants, fixed at compile time |

The defaults of the tunings and knobs are in the code; the python
configuration only overrides the tunings. The knobs are constants so that there
is no mutable global state and switched-off code compiles away. When an
iteration needs its own value of a knob, the knob moves to the tunings.

Outside CMSSW, in the standalone build, the knobs are variables, and both
tunings and knobs can be set from the `mkFit` command line for scans.

## Layer processing

**Code:** `process_layer()`, `process_layer_batch()`, the `rk_*()` functions,
`select_completed_ccreps()`.

The overview at the top of `MkFinderV2p2.cc` follows one seed through one
layer. This section explains how the code is organised around that, and why.

**Full vector lanes.** The track math runs on Matriplex batches of NN lanes (8
with AVX, 16 with AVX-512), one track per lane. One seed rarely brings NN
candidates into a layer, so candidates from many seeds enter the layer together
and go through propagation, search window and hit pre-selection NN at a time, in
layer batches. Pre-selection uses a second batch shape: candidates bring
different numbers of hits, so it runs one lane per (candidate, hit) pair, filled
across candidates.

**Kalman updates across batches.** In the in-layer search each accepted hit can
be followed by more hits in the same layer, so the number of Kalman updates a
candidate needs is not known in advance (see "In-layer combinatorial search").
The updates are therefore queued, and a Kalman batch runs only when NN updates
are waiting, or at the end of the layer. The first hit of a path in the layer
and the later ones are queued separately. The first can use the crossing with
the module plane found in pre-selection; a later one has to find the crossing
from the state after the previous hit. Lanes are independent, so the result is
the same as running each step of each batch on its own, up to rounding.

Measured on T100, on a Xeon Gold 6130 (Skylake-SP), against Kalman batches run
as soon as each layer batch had produced them: the filled lanes went from 6.19
to 7.70 of 8 for first hits and from 4.38 to 6.99 of 8 for later hits. The
forward search became 5.3 % faster with `-mavx`, 4.6 % with AVX2 and 3.8 % with
AVX-512.

**Finishing a seed early** (on by default). Because batches mix seeds, the work
of one seed ends at different times. For each seed the finder counts the
candidates still waiting for pre-selection, and the Kalman updates and paths
still pending. When both counts are zero, the seed's paths compete and its new
candidates are chosen right away instead of at the end of the layer, and the
memory of its paths is reused. This changes neither the result nor the time
measurably; it lowers the paths held in memory per layer from 50 nodes to 12 on
average. It is on because it costs nothing and bounds the memory a finder needs
by the seeds in flight rather than by all seeds of the layer.

## Candidate pickup and stopping cuts

**Code:** `awaken_candidates()`, `begin_next_Ccrep_in_layer()`,
`stop_cuts_at_pickup()`, `fake_hit_index()`, `end_layer()`.

**Pickup.** Seeds end at different layers. A seed waits until the plan reaches
the entry that holds the layer of its last hit, and is searched from the next
entry on.

**Stopping cuts.** When a candidate enters a layer, two cuts that need only its
own state are applied before any work is spent on it:

- pT below the iteration's `minPtCut`;
- the looper stop, in the forward search only: pT below 1.2 GeV, transverse
  radius above 25 cm, and the angle between the transverse position and the
  transverse momentum above 78.5 deg (`Policy::looper_max_pt`, `looper_min_r`,
  `looper_max_angle`, pi/2 - 0.2 rad). Past that angle the track crosses
  modules at grazing incidence, the clusters get wide, and further hits degrade
  the measurement. The test is equivalent to cos(momPhi - posPhi) < sin(0.2),
  so it does not depend on where phi wraps around.

A stopped candidate records why: minimum pT, looper, or too many holes. The
reason is not yet exported to CMSSW.

**Holes.** A candidate that takes no hit in a layer records a hole. The
iteration's hole limits (`maxHolesPerCand`, `maxConsecHoles`) decide whether it
is a miss or a stop, as in V1. If the WSR verdict (see
[Within-sensitive-region verdict](#within-sensitive-region-verdict)) is edge or
gap, the hole is recorded as that kind instead, and does not count against the
limits.

**Retiring.** A CombCandidate is finished once every candidate under it has
stopped.

## Layer crossings and the Hermite cubic

**Code:** `MkBins::prop_to_limits_in_order()`, `prop_to_layer_edges()`,
`preselect_hit_batch()`, `mini_propagators::Hermite3D` and `Hermite3DOnPlane`.

Each candidate is propagated to the layer's two bounding surfaces (cylinders in
the barrel, planes in z in the endcap) in the order the track meets them: the
entry crossing and the exit crossing. These propagations carry only the track
parameters, not the covariance, and are in closed form.

Inside the layer the trajectory is modelled by a cubic Hermite curve through
the two crossings, with their momenta as tangents. For each hit read from the
window the cubic is solved onto the hit's own module plane with two Newton steps,
and both the cut and the state handed to the Kalman update are taken at that
point. The exact helix-plane intersection is transcendental for a tilted module;
the cubic is what makes a crossing per hit affordable. On a plane of constant z
the first step is exact, since z is linear along the cubic. After the two steps
the point lies within 10 nm of the module plane for 99.8 % of the solves (forward
search, 10 ttbar PU200 events; 93.6 % after one step, with a tail of up to
millimetres in the 2S layers). Against the exact helix crossing, the point is off
across the strip by 2.1 um in the PS layers and 2.9 um in the 2S layers at the
99th percentile, the cubic's own model error, and by 0.08 um in the endcaps. The
tail, 2.9 mm at the 99.9th percentile in the 2S layers, comes from low-pT tracks
that cross the modules at grazing incidence, where the cubic is a poor model of
the helix.

## Reachability

**Code:** `mini_propagators::propagate_to_r()`, `kReachMargin`, `kReachTolerance`.

In the transverse plane the track is a circle of radius R_c whose centre is at
distance d_c from the beam line, so it reaches only radii in
[|d_c - R_c|, d_c + R_c]. At both ends of that band the track grazes the
cylinder (dr/ds = 0) and the crossing solve degenerates. The target radius is
therefore kept `kReachMargin` = 0.1 cm inside the band. A target outside it is
clamped to the band edge, which returns the point of closest approach on the
trajectory, and marks the propagation as failed, which the WSR verdict then
reads (see [Within-sensitive-region verdict](#within-sensitive-region-verdict)).

The margin is provisional. It was chosen on 10 events of the inward search. One
length for the whole detector is probably the wrong shape. Whether a track
reaches a layer is a statement about a track with an error, so the band should
rather scale with the track's expected q error, per layer. A retuning needs
outward searches, pixel seeds and the full pT range.

## Within-sensitive-region verdict

**Code:** `MkFinderV2p2::determine_wsr()`, `CandIncoming::m_wsr`.

Before any hit is looked at, each candidate gets a verdict on whether its track
crosses the layer's sensitive area: **inside**; **edge**, it may have passed
beside the modules; or **outside**, it does not cross the layer, which skips
the layer. An outside verdict where a disc has an opening is marked as a
**gap**, and records a gap hole, since the track did pass through the disc. An
edge or gap hole does not count against the hole limits. V1 tests one
propagated point; v2p2 has both crossings of the layer's bounding surfaces and
tests the segment between them.

- In radius: if the propagation reaches neither surface, the track turns back
  before the layer: outside. If it reaches only one, the track enters and turns
  back inside the layer: at best edge. In the endcap the surfaces are planes in
  z, which the track always reaches, and the q test decides alone.
- In q: the layer's q limits are moved by 5 sigma of the track's q error
  (`Policy::wsr_n_sigma`). The candidate is outside if its segment lies wholly
  beyond the limits moved out by that amount, inside if it lies wholly within
  the limits moved in by it, and edge otherwise. Erring towards edge is the safe
  side: a real miss recorded as edge is not counted against the limits, while
  an inside that should have been edge charges a hole to a candidate that
  crossed nothing.
- A disc with an opening, an r range without modules: a segment wholly inside
  the opening is outside, marked as gap; one that touches its edge is edge. No
  phase-2 layer declares an opening today.

Skipping layers the track does not cross matters because the plans are
inclusive: a plan for the transition region lists every layer any track there
may cross. Without the verdict each of those layers would record a hole.
Measured on 20 events of the forward search: 64.5 % of layer searches in the OT
barrel and 23.8 % in the disc layers are judged outside and skipped, and the
sim track has a hit in the skipped layer in 0.05 % and 0.30 % of those.

## Hit extents

**Code:** `LayerOfHits::HitInfo`, `hit_phi_half_extent_of()` and
`hit_r_half_extent_of()` in `HitStructures.cc`.

For each hit the layer caches the extents that the window and the pre-cut use.
All are `hl_fac` times a sigma from the hit's own covariance, with `hl_fac` = 3
for pixels (a Gaussian 3 sigma) and sqrt(3) for strips, where a crossing uniform
along a segment of half-length L has sigma = L/sqrt(3).

- **q half-length**: `hl_fac * sqrt(ezz)` in the barrel. For a strip along the
  unit vector u the covariance is (L^2/3) u u^T, so this is L |u_z|, the strip's
  projection onto z, and the module tilt is already included. That is why the
  tilted OT PS barrel shows medians well below the nominal half-lengths (see
  "Two factors per coordinate" in [Search window](#search-window)). The P
  macro-pixels count as strips here, with `hl_fac` sqrt(3). In the OT 2S
  barrel, where the modules are flat, the median on the I50 events is 2.5125 cm
  against a nominal 2.5. In the endcap it is `hl_fac * sqrt(exx + eyy)`, with
  exx + eyy = sigma_r^2 + r^2 sigma_phi^2. That is right for radial strips and a
  factor sqrt(2) too large for endcap pixels, where sigma_x and sigma_y are
  similar; their q window is wider than needed, which costs scanned hits, not
  found ones.
- **phi half-extent**: from sigma_phi^2 = (y^2 exx - 2xy exy + x^2 eyy) / r^4,
  the across-strip error seen from the beam line. It needs no barrel/endcap
  branch. For a 2S strip it is `hl_fac * pitch / sqrt(12)` / r, 45 um / r for a
  90 um pitch, which matches the 4e-5 rad measured in the OT 2S barrel.
- **qbar half-extent**: `hl_fac * sigma_r` in the barrel, 0 in the endcap, used
  by the line pre-cut for tilted strips.

## Search window

**Code:** `MkBins::determine_bin_windows()`, `MkBins::find_bin_ranges()`,
`MkFinderV2p2::preselect_hit_batch()`, `surface_referenced_dq()`,
`Config::V2p2::Window`.

The window decides which hits of a layer a candidate looks at. It has two
parts: a **fetch**, a range of (phi, q) bins read from the binnor, and a
per-hit **cut** applied to the hits found there. The fetch must cover
everything the cut accepts: a hit the cut would accept but the fetch never read
is lost without any sign.

### The cut

```
|dq|   < dq_trk_fac   * T_q   + dq_hit_fac   * H_q
|dphi| < min(dphi_trk_fac * T_phi + dphi_hit_fac * H_phi, dphi_max_rad)
```

dq and dphi are the residuals between the hit and the track's crossing with the
hit's own module plane. T_q and T_phi are three sigma of the track's position
error at the layer, in q and in phi; T_q is referred to the hit's module
surface (see "dq surface reference" below). H_q and H_phi are the hit's own
half-extents (see [Hit extents](#hit-extents)). The factors are 1.5 and 1.2 in
q, 2 and 3 in phi, so the track terms are 4.5 and 6 sigma. In the code T_q is
`dq_trk`, made from `dq_track` at fixed path length, T_phi is `dphi_track`, and
H_q and H_phi are `hit_q_half_length` and `hit_phi_half_extent`.

**The phi margin is capped at `dphi_max_rad` = 0.05 rad**, in the fetch, the cut,
the line pre-cut (times `dphi_slack`) and the density region. Without it the
margin reached 1.34 rad: low-pT candidates carried over long steps at high
|eta|, and loopers of a few tens of MeV in the backward search. No hit further
than 0.031 rad in phi from the track ends on a final track, and the
pre-selection queue keeps the hits closest in phi anyway. Over 100 PU200 events
the cap changes found, fake and duplicate tracks by nothing and the reco tracks
by one. With the crossing span at most pi, a window is at most pi + 0.1 rad
wide. The phi bins are walked from p1 to p2 on the masked axis, which cannot
tell a window over the full turn from an empty one, so the knob must stay below
1.5.

**Two factors per coordinate.** Which term dominates depends on the layer. The
median q half-length of a hit (on the I50 events) is 0.0075 cm in the pixel
barrel, 0.042 cm on the P sensors and 0.80 cm on the S sensors of the OT PS
barrel, and 2.51 cm in the OT 2S barrel, while the track term does not vary like
that. In strips the window is set by the strip length, in pixels by the track's
error. One factor on the sum could not be right for both, and a scan of it could
not say which term it was moving.

`dq_hit_fac` multiplies the hit's own half-length, so 1.0 is a hard floor: below
it the window no longer covers the whole strip. The default 1.2 keeps 20 %
margin. Going from 1.8 to 1.2 changes nothing significant (F30 -3 found tracks
and -15 fakes, I50 +55 recovered hits and -2 fully recovered tracks), so the
narrower window is kept.

### Track covariance at the layer

The track terms need the track's position error at the layer. It is obtained by
transporting the position part of the covariance from the previous hit along the
helix to the layer's far crossing, in closed form from quantities the
propagation to the layer already has. The derivation is in [Transport of the
position covariance](#appendix-transport-of-the-position-covariance). No
material is added here. mkFit applies a layer's material as a thin scatterer at
the hit, which changes only the direction terms, so the material of the step
does not reach the position within it; what is already in the starting
covariance is carried along. Where that material is underestimated, the
transported error is too small as well (see the phi factor below).

Against a full helix-to-plane propagation with uniform field, over 200508 lanes,
the sigmas agree to 7e-5 relative and the correlations to 1.6e-4 at the worst
lane. With the parametrised field at the starting point they differ by 0.4 % at
the 99th percentile. The transport uses the same uniform field as the
propagation to the layer and the Hermite cubic.

**dq surface reference.** The covariance is transported to a fixed path length,
not to a plane. Its q error says where the track is after a given distance, not
where it crosses the module, and the two differ by how far the crossing can
slide along the track. Sliding each point along the momentum direction p^ until
it meets a surface with normal n^ is a linear map on the position error,

```
dx_s = (I - p^ n^T / (n^.p^)) dx
```

so the q variance becomes v^T C v with v = e_q - ((e_q.p^)/(n^.p^)) n^. The cut
applies it with n^ the hit's own module normal. For a cylinder normal and a
radial track this multiplies the q error at fixed path length by
1/sin^2(theta). With the module normal
the correction is almost nil for the OT PS barrel modules, which face the
interaction point, and tilted and flat layers use the same formula. Ratio of the
measured residual width (the core of the distribution, without its tails) to
the quoted sigma_q, 20 events of the inward search, pixel barrel:

| \|eta\| | 0-0.8 | 0.8-1.6 | 1.6-2.0 | > 2.0 |
|---|---|---|---|---|
| without the reference | 1.64 | 4.66 | 17.0 | 46.0 |
| with the reference | 1.38 | 1.43 | 1.71 | 2.14 |

**phi error.** `sigma_phi` comes from the xy covariance through J = (-y, x)/r^2,
so |J| = 1/r. One window covers the whole layer, so J is taken at the crossing
with the smaller radius, where `sigma_phi` is largest. At pixel layer 0 the
outer radius is 1.33 times the inner one.

**`dphi_trk_fac` = 2.** The hit's own phi extent is tiny, about 4e-5 rad in the
OT 2S barrel, so the phi cut is carried by the track term, and that needs a
factor 2. The per-hit extent replaced a flat tolerance of 0.0246 rad, which was
half a phi bin rather than a property of the hit; with the factor 2 it gives the
same counts. Differences against the flat tolerance:

| `dphi_trk_fac` | F30 found | F30 fakes | I50 fully recovered tracks |
|---|---|---|---|
| 1.0 | -41 (4.2 sigma) | +87 | -582 |
| 1.5 | -2 | +10 | -29 |
| 2.0 | +1 | +2 | identical |

Above |eta| 0.8 the factor also makes up for a phi covariance that is too small.
The ratio of the measured phi residual width to the quoted `sigma_phi` (the phi
pull), 10 events of the forward search, is 1.25 at |eta| < 0.8, 2.20 at 0.8-1.6,
3.40 at 1.6-2.0 and 2.66 above. The large values are attributed to the material
description in the transition and endcap regions; a smaller deficit, the 1.25 in
the central barrel, is present everywhere and not yet explained. When the
covariance is right, the factor should come down.

### The fetch

The fetch is computed from the cut (`find_bin_ranges()`), with two differences.
The cut is per hit, and the fetch is made before any hit is known, so it uses
the layer's largest hit extents. And the fetch uses the track's q error at fixed
path length, before the surface reference, so at high |eta| the cut can reach
past it; the spare q bin below covers that. The phi range is an arc on a circle:
it must include the bin holding its upper edge, and a half-width of pi or more
must give the full circle, not a small arc.

Whole spare bins can be added beyond the cut:

- phi: none (`Window::phi_extra_bins` = 0). Against one spare bin, F30 -1 found
  track and I50 +1 recovered hit, and every fetched hit costs work.
- q: one (`Window::q_extra_bins` = 1), for the surface reference. Without it I50
  loses 1373 of 43358 fully recovered tracks, all in pixel-barrel hits of tracks
  that also cross the discs; F30 is unaffected. With the line pre-cut in, a
  spare bin costs little.

## Line pre-cut

**Code:** `MkFinderV2p2::select_hits()`; switches and slack factors in
`Config::V2p2::PreCut`; the per-hit input `LayerOfHits::hit_qbar_half_extent()`.

**What it does.** Every hit read from the window's bins costs a crossing with
its own module plane and a lookup of the module geometry before the real cut
can be applied. The pre-cut rejects most of them earlier, from a cached array
of hit positions and extents and four numbers per candidate: the offset and
slope of a line in q and in phi.

Between its two crossings of the layer's bounding surfaces the track is taken
as a straight line in (qbar, q) and in (qbar, phi), with qbar the radius in the
barrel and z in the endcap. The line is evaluated at the hit's own qbar, and the
hit is dropped when its q or phi is further from the line than a tolerance
looser than the real cut.

**Tolerances**, with g the slope of the line in q and g_phi in phi:

```
q:   dq_slack   * dq_trk_fac * dq_track * (1 + g^2) + dq_hit_fac * hit_q_half_length
       + qbar_fac * |g| * hit_qbar_half_extent
phi: dphi_slack * min(dphi_trk_fac * dphi_track, dphi_max_rad) + (hit term of the real cut)
       + qbar_fac * |g_phi| * hit_qbar_half_extent
```

with `dq_slack` 2, `dphi_slack` 1.5 and `qbar_fac` 1.2. The tolerances are
written in terms of the real cut's own factors, so the pre-cut stays looser
when those change.

**Why each term.**

- `(1 + g^2)` equals 1/sin^2(theta) in the barrel and 1/cos^2(theta) in the
  endcap, for a track from the origin. In both it refers the track's q error to
  the layer surface, as the real cut does per hit. Without it
  the pre-cut would drop hits that the real cut accepts at high |eta|.
- `hit_qbar_half_extent` covers tilted strips. A hit's qbar is its centroid
  radius, but on a tilted module the track crosses the strip somewhere along
  its length, and the strip has a radial component, so the radius is uncertain
  by up to L sin(tilt). Through the slope of the line that becomes a shift in q
  and in phi. In the endcap qbar is z, where this does not arise, and the term
  is skipped.

**Offline estimate.** Replaying the pre-cut over recorded searches counts, for
each candidate-hit pair, whether the pre-cut would drop it and whether the real
cut accepts it. A false reject is a pair the real cut accepts and the pre-cut
drops. Forward 5 events, inward 10 events:

| | forward: pairs | rejected | false rejects | inward: pairs | rejected | false rejects |
|---|---|---|---|---|---|---|
| q only, slack 2 (first version) | 11.9 M | 79.3 % | 82 | 1.83 M | 85.8 % | 0 |
| q slack 2 and phi slack 1.5 (default) | 11.9 M | 89.8 % | 405 | 1.83 M | 89.1 % | 15 |
| q, slack 2, without the surface factor | 11.9 M | 80.7 % | 17610 | 1.83 M | 89.0 % | 10381 |
| q, slack 1 | 11.9 M | 82.2 % | 2278 | 1.83 M | 89.7 % | 554 |
| q, slack 1, without the qbar term | 11.9 M | 84.3 % | 87659 | 1.83 M | 90.1 % | 3364 |

The real cut accepts 8.2 % and 7.6 % of the pairs. All the q false rejects
without the qbar term are in the tilted OT PS barrel layers. The phi false
rejects sit mostly in the OT 2S barrel.

**Physics.** Against the same configuration with the pre-cut off, F30 found
tracks, fakes, duplicates and the momentum-resolution width are identical to
the unit, and so are the I50 counts and the truth-matched hit efficiencies, with
q alone and with q and phi. So none of the falsely rejected hits was one a final
track would have taken.

**Time.** T100, 2026-09: 47.95 s without the pre-cut, 35.36 s with q, 33.81 s
with q and phi; V2 took 62.95 s on the same events. In one forward event the
pre-cut rejects 2.87 M of 3.18 M scanned pairs (90.4 %); of the 306 k left, 79 %
pass the real cut.

## Reduction and hit ordering

**Code:** `CandIncoming::m_pqueue`, `preselect_hit_batch()`,
`order_hits_along_path()`.

**Reduction.** Of the hits that pass the cut, a candidate keeps the
`max_presel_hits` (6) with the smallest phi residual. phi is the coordinate in
which strips measure; q is mostly the strip length. The cap bounds what the
in-layer search can see. How often modules overlap varies strongly across the
detector, so the cap should become per layer. In a paired plan entry each sensor
keeps its own 6, so a busy sensor cannot crowd out the other.

**Ordering.** The kept hits are then put in the order the track meets them,
which is the order the in-layer search walks them in. The key is the turning
angle dalpha from the candidate's state to the hit's crossing. For all hits of
one candidate it grows with the path length, since they are reached from the
same state with the same curvature. dalpha is signed, and its sign depends on
the charge, so the key is dalpha times the sign of its change from the layer's
entry crossing to its exit crossing; that gives path order for both charges and
both search directions. The phi residual decides which hits survive; path order
decides the order they are walked in. The order is computed once, from the
candidate's incoming state. Each update along a path moves the trajectory by
about the hit resolution, so recomputing the order after it could only swap
hits that close together.

Two hits on one module share the crossing with its plane and so have equal
keys. Which comes first does not change which paths exist, since a path takes
at most one hit per module; a fixed order (larger phi residual first, then by
hit index) only makes the result reproducible.

**Between the two sensors of a module.** In the OT PS barrel the tilted modules
put the partner sensor at larger or smaller radius module by module, so the
order in which a track meets P and S cannot be read from the layer number. The
split plan nevertheless searches them in plan order, P first; the hits of the
two sensors then enter the track in plan order, not always in path order. The
backward fit puts them back in path order (see [Backward fit](#backward-fit)).

## In-layer combinatorial search

**Code:** `expand_in_layer()`, `harvest_nodes()`, `CandNode`,
`MkFinderV2p2::m_node_arena`.

**Why.** Within one mkFit layer a track can cross more than one module where
modules overlap. Per OT layer crossed (a mkFit layer, one sensor layer), on 30
events of the forward search with truth-matched hits (2026-09), the sim track
has hits in two modules in 21.2 % of crossings and in three or more in 3.1 %.
v2p2 takes two in 19.8 % and three or more in 1.5 %; production mkFit in CMSSW,
which stops at two, takes two in 20.9 % and three or more in 0.02 %. More than
the count, the search keeps the alternatives: each hit, each set of hits, and no
hit compete at the end of the layer on one score, so a well-fitting wrong hit
does not have to be accepted before a better path is seen.

**What it does.** A path through the layer is a subsequence of the candidate's
kept hits, in path order. Per candidate, the search is

```
frontier = { kalman(state, h) : h in hits }                       # depth 0
for depth = 1 .. max_hits_per_layer - 1, while frontier is not empty:
  next = { kalman(n.state, h) : n in frontier,
           h after n's hit in hits, h not on a module of n's path }
  frontier = next
```

where `kalman()` keeps an outcome only if the hit passes the acceptance cut
([Kalman hit acceptance](#kalman-hit-acceptance)). Every path of every depth
enters the end-of-layer selection. The code computes the same paths in a
different order, to keep the vector lanes full (see "Kalman updates across
batches" in [Layer processing](#layer-processing)).

The first hit of a path starts from the candidate's state, at the crossing the
Hermite cubic has already found, so the propagation is handed the path length.
Later hits start from an updated state, for which no crossing is known, and the
propagation to the module plane solves for it. A Hermite cubic built from the
updated state alone would be cheaper; it is not used yet.

Walking an ordered list forward only reaches every subset of the hits exactly
once. Taking both hits of an overlap therefore needs no special case, and there
is nothing to de-duplicate. Path order also puts the Kalman updates in the
order the track meets the hits.

A second hit on a module the path already has a hit on is not taken. It is a
split cluster or another track's hit, and a split cluster is one measurement:
taking it twice shrinks the covariance without adding information.

`max_hits_per_layer` (4) caps the hits per path. Forward, on 10 events, found
tracks stop changing at 3. Inward on 20 events, fully recovered tracks go from
82.4 % at 2 to 90.8 % at 4.

**Effect.** F30, beam width 3, against the first version's one best hit per
layer, in quality-val's counts, which reward the longer tracks the search
produces (the MTV comparison of v2p2 with V2 is in [Layer-step
score](#layer-step-score)): found tracks 18455 to 18846, fakes 3217 to 2765.
Truth-matched hits per found track go from 10.78 / 11.55 / 11.32 to 12.70 /
14.32 / 14.29 in barrel / transition / endcap. Inward on 20 events, at most 2
hits per path: pixel hits recovered 75.7 % to 92.2 %, fully recovered tracks
52.8 % to 82.4 %. A beam width of 3 is the default with the search on; from 3 to
6 F30 gains 39 found tracks.

**Storage.** A path is stored as a chain of nodes, one per hit, each with the
state after its update and the index of the node before it; a surviving path is
read by walking back from its last node, and nothing walks forward. All nodes of
a finder sit in one vector, because one Kalman batch draws lanes from many
seeds. The vector is cleared at the end of each layer and keeps its capacity. A
seed's nodes are reused only once the seed has been selected, never earlier,
since a node may still have children waiting. A node carries a full track state
and lives for one layer; the hits a track keeps for the event are stored
separately and compactly.

## Kalman hit acceptance

**Code:** `plane::chi2_scaled()` in `KalmanOpArgs::do_kalman_stuff()`,
`harvest_nodes()`, `V2p2Params::chi2_trk_fac` and `hit_chi2_cut`.

**The cut.** A hit is accepted onto a path if

```
chi2_acc = r^T (f^2 * C_trk + C_hit)^-1 r  <  hit_chi2_cut
```

with r the residual, C_trk and C_hit the track and hit covariances, all in the
module's local 2-D frame, and f = `chi2_trk_fac`. The defaults are f = 3 and a
cut of 9.21, the 99 % point of chi2 with 2 degrees of freedom. Only this
accept/reject decision uses the scaled covariance; the Kalman update and the
score use the plain chi2.

**Why two factors.** The cut used to be the plain chi2 below a constant 30. That
one number stood for two things: a probability, and compensation for a track
covariance that is too small, by 1.2-2x in sigma on the inward search and more
above |eta| 1.6. The hit covariance comes from the cluster and matches the
sensor pitch, so it is taken as it is. The factor f on the track covariance
carries the deficit, and the cut is left as a probability. The deficit is on the
track side: the improvement from the factor is flat in pT above 0.9 GeV, so it
is not multiple scattering, and scaling the hit covariance instead loses badly
(below). The search window is split the same way, see [Search
window](#search-window).

**What it changes.** With MTV's association, 20 events, f = 3 and 9.21 against
the plain chi2 < 30 give -32 found tracks and -4 fakes: no measurable change.
The cut is kept as the principled replacement of the fixed 30, not for a gain;
what removed v2p2's extra fakes is the score (see
[Layer-step score](#layer-step-score)). The scan on F30, against the plain
chi2 < 30, in quality-val's counts, which reward track length:

| cut | chi2_trk_fac | d found | d fakes |
|---|---|---|---|
| 20 | 1 | -129 | +322 |
| 13.8 | 1 | -382 | +781 |
| 9.21 | 1 | -953 | +1612 |
| 30 | 1.5 | +110 | -225 |
| 20 | 2 | +115 | -233 |
| 13.8 | 2 | +95 | -187 |
| 9.21 | 2 | +19 | -44 |
| 13.8 | 3 | +118 | -242 |
| 9.21 | 3 | +115 (8.3 sigma) | -231 (12.3 sigma) |

In these counts, lowering the plain cut loses tracks and adds fakes at every
value, and with the track factor the counts level off at f = 2-3. The change
comes from |eta| 1.5-2.5, where the phi pull is furthest from 1. A lower cut
also shortens tracks, which moves these counts by itself, so this supports the
compensation reading rather than proving it. The width of d(pT)/pT on the tracks
found in both does not move (0.02078 to 0.02085). On I50: pixel hits recovered
198340 to 200841 (93.8 % to 95.0 %), fully recovered tracks 39167 to 39798 (90.3
% to 91.8 %).

**Time.** T100, 2026-09, on a Xeon Gold 6130 (Skylake-SP, `-mavx`), with F30
against the plain chi2 < 30. f = 1.0001 computes the second chi2 without
changing any decision, which isolates its cost:

| cut | f | d found | d fakes | time [s] | note |
|---|---|---|---|---|---|
| 30 | 1 | | | 22.62 | |
| 30 | 1.0001 | 0 | 0 | 22.66 | second chi2 only |
| 13.8 | 1.5 | -5 | +19 | 22.60 | same physics |
| 9.21 | 2 | +19 | -44 | 22.70 | +0.3 % |
| 9.21 | 2.5 | +94 | -187 | 22.97 | +1.6 % |
| 9.21 | 3 | +115 | -231 | 23.14 | +2.3 % |

The second 2x2 chi2 costs 0.2 %. With unchanged decisions the time is unchanged:
every pre-selected hit goes through the Kalman update either way, and the cut
only decides which results become nodes. Time grows with the true hits kept.
What reaches the Kalman update is set upstream, by the window and the line
pre-cut.

**Tried and not taken.**

- f as a function of |eta|, a ramp from f_lo below 0.8 to 3 above 1.6. f_lo of
  1 / 1.5 / 2 loses 245 / 76 / 24 F30 found tracks and 2240 / 538 / 164 I50
  fully recovered tracks. The central region needs the full factor too,
  although its phi pull is only 1.25: the deficit there is the smaller one
  that is not yet explained (see [Search window](#search-window)).
- A separate f for the second and later hits of a path: f 1.5 changes nothing
  forward and loses 53 fully recovered tracks on I50, and saves no time, since
  the cost is set by what the first hit accepts.
- Scaling the hit covariance instead, h^2 C_hit (equivalent to f = 1/h at a cut
  of h^2 * 9.21): h = 2 loses 776 F30 found tracks and 6601 I50 fully recovered
  tracks.
- f of 4 and 5: within noise of 3 in both directions, so one value serves
  forward and inward.

f = 3 is a compensation and should come down when the track covariance is
fixed, as `Window::dphi_trk_fac` should.

## Grazing incidence

**Code:** `Window::min_cos_incidence` (`src/V2p2Config.h`) and its cut in
`preselect_hit_batch()`; the chi2 >= 0 test in `harvest_nodes()`; the covariance
update in `plane::update()` (`src/PlaneSteps.h`).

**Why.** In float, the Kalman update breaks the track covariance when the
track's local variance across the strip is much larger than the hit's.
`plane::update()` forms I - KH and takes (I - KH) C. Across the strip K is close
to 1, and 1 - K keeps only what float precision leaves of that subtraction.
Above a ratio C/V of about 1e6 between the track's and the hit's local variance,
the updated variance across the strip has a random sign. Such ratios come from
grazing incidence: the track's transverse position error projects onto the
module plane as 1/cos^2 of the incidence angle. The next hit of the same layer,
on the neighbouring module, then starts from a covariance that is not positive
semi-definite. Its residual covariance is not positive definite, and its chi2 is
negative. Before this section's changes a negative chi2 passed the acceptance
cut and raised the layer-step score, which subtracts chi2 / 2, so the most
broken path won. The rounding of a partially filled batch decides the last
digits, so the same seed could end on different hits depending on how the seeds
were chunked.

Measured on 20 D121 PU200 events, 10.98 M search updates, each replayed bit for
bit in float and redone in long double (`mkfit-results/core/2026-11-16/negchi2/`
in the maintainer's records):

- 45 updates have chi2 < 0, from 12 seeds in 9 events. 44 are forward, on 2S in
  layers 12, 13 and 15; 1 is in the backward search, on pixel layer 1. Each is
  the second or a later hit of a path in its layer.
- In 48 of 50 (two seed-loop grains), the covariance was broken by the update of
  the previous hit of the same layer. Propagation and material never broke a
  positive semi-definite covariance, and the chi2 and the 2x2 inverse were right
  on every valid prediction.
- 2S updates on a valid prediction that leave a covariance that is not positive
  semi-definite, by |cos| of the incidence:

| \|cos\| | updates | broken by the update |
|---|---|---|
| < 0.1 | 1513 | 301 |
| 0.1 - 0.3 | 22804 | 51 |
| >= 0.3 | 1.56 M | 0 |

- The relative error of the updated variance across the strip grows about
  tenfold per decade of C/V: 1e-6 at 10, 7e-4 at 1e4, 0.2 at 1e6.

**The cut.** A hit is not taken when the track crosses its module at |cos| of
the incidence below `Window::min_cos_incidence`, which is 0 (off) by default; at 0
the |cos| computation is skipped. |cos| is |p.n| / |p|, from the
Hermite state on the module plane and the module normal. The cut also has a
reason of its own: by geometry, a 2S strip hit at |cos| 0.1 spreads over about
3 mm across the strips (sensor thickness ~0.3 mm times tan of the angle), some 30
strips, while its error is the pitch over sqrt(12). The same code runs inward
and outward, so the cut applies to both. Separately, `harvest_nodes()` rejects
every update with chi2 < 0 or chi2_acc < 0, whatever the cause. The backward
search also breaks covariances on the TFPX discs at |cos| 0.5 - 0.66, from a poor
prediction (24 in the 20 events, with no negative chi2 following), and the
incidence cut does not reach those.

**Measured, 100 events** (D121 PU200, MTV association, paired by event against
the code without this section's changes; `mkfit-results/core/2026-11-16/grazing-cut/`
in the maintainer's records):

- The chi2 < 0 rejection alone finds 22 sim tracks more, all low-pT barrel
  tracks (|eta| < 0.79, pT 0.49 - 0.65 GeV) that no reco track matched before:
  +20 +- 4 at pT 0.5 - 0.9. Efficiency at pT > 0.9, fakes, duplicates, hits per
  track and the pT resolution do not change.
- The cut at 0.05 or 0.1 adds nothing measurable on top of the rejection. At 0.1
  the updates with a negative chi2 go from 224 to 24, all rejected, and the cut
  takes 0.24 % of the forward 2S hits and 0.9 % of the backward search's
  pixel-barrel hits.
- At 0.2 and 0.3 the cut takes 72 % and 99.6 % of the backward search's
  pixel-barrel hits. On a flat barrel module |cos| is at most sin(theta), 0.16 at
  |eta| 2.5, so a cut on the total incidence removes the pixel-barrel hits of
  every track above |eta| of about 2.3, however straight it crosses in phi.
  Endcap tracks lose 0.07 and 0.18 hits each, duplicates rise by 0.09 and 0.35
  points, low-pT fakes by 23 and 30 tracks.
- Hence the default 0: the rejection stays, the cut is off. A cut on the
  across-strip component of the incidence, or on strip layers only, would leave
  the pixel barrel alone; not built.

**Loopers.** A looper meets grazing incidence where it turns: at its turning
radius its momentum is perpendicular to the radius. One of the broken updates
above is such a turn, at r = 109.6 cm, within 1.3 - 1.8 degrees. The cut takes
those hits away, and a looper's candidate collects holes at its turn, which can
stop it. What a looper needs at a grazing encounter is to pass through with its
state changed as little as possible: the hit acknowledged as present, the energy
loss of the crossing applied, and no position update. That needs a hit index
code of its own next to `kHitMissIdx` and the others in `Hit.h`, for a hit that
is present, not used in the update, and not counted as a hole. Not built.

**The update itself.** The fix of the update is the Joseph form,
C' = (I - KH) C (I - KH)^T + K V K^T, which is symmetric and positive
semi-definite by construction. In the replay (float, written in the replay
macro, not library code) it kept every updated variance positive, with relative
error below 1e-5 up to C/V of 1e6 and a median of 1e-5 at 1e7. The final fit
already updates in this form, vectorised (`update_joseph()` in `FinalFit.cc`).
`plane::update()` is shared: `kalmanOperationPlaneLocal()` calls it for the clone
engine, V2 and the backward fits. Switching it to the Joseph form changes those
too, so it is not done here. With it, the incidence cut would be about the hit's
quality alone. The fix has two parts: this stable form, and the processing of a
grazing hit described under "Loopers".

## End-of-layer selection

**Code:** `select_and_materialise()`, `offer_best_short()`, `SelEntry`.

At the end of a layer everything that could continue a CombCandidate competes
in one list, on one score:

- every path of every candidate that searched the layer;
- every such candidate again, as the hole it would record, also when it found
  paths;
- every candidate that did not search the layer (stopped, or skipped when
  entering it), at its unchanged score.

The best `maxCandsPerSeed` survive. One list is possible because the score adds
up over layer steps ([Layer-step score](#layer-step-score)), so paths from
different candidates are on one scale. It runs at the end of the layer, or as
soon as all of the seed's work in the layer is done (see
[Layer processing](#layer-processing)), because one seed's candidates can be
spread over several batches.

**The hole competes with the hits.** A candidate may prefer a well-fitting wrong
hit to a hole, and no score of a single hit can tell. Offering the hole as a
competitor defers the decision until the paths can be compared. A candidate
outside the layer has no hole and competes unchanged.

**Two options, measured and left off.** Both keep an alternative alive in case
a hit taken earlier was wrong.

- **Reserve a hole slot** (`InLayer::reserve_hole_slot`): always keep the best
  candidate that took no hit in this layer among the survivors. When every
  survivor took a hit, none allows for that hit having been wrong. Measured: a
  difference of 3 found tracks forward and +25 recovered hits inward, so no
  change.

- **Best short** (`InLayer::best_short`, outward only): keep the best stopped
  candidate aside, as V1 does, and put it back at the end of the search if it
  still wins. Measured: -9 found tracks, no change. v2p2 needs it less than
  V1, since a stopped candidate already stays with its seed and competes at the
  end. Inward, a candidate that stops early lacks the part of the track nearest
  the beam, which is what the inward search is for, so it is not kept.

Neither can help while nothing recognises a branch that went wrong: the first
wrong hit of a diverged candidate fits at a chi2 of about 3.75, among the
correct hits.

Survivors are built as copies before the CombCandidate is changed, because
several survivors can descend from one candidate.

At the end of the search each seed's candidates are rescored with a whole-track
scorer, which picks the seed's best candidate: outward the final-pick scorer,
inward the scorer the iteration names for its search. That score is what the
output and the duplicate removal see. With `v2p2:llh` it is built from the
layer-step score, see "The final pick" in [Layer-step score](#layer-step-score).

## Layer-step score

**Code:** `V2p2Score.h`, `LayerStepFeatures`, `Config::V2p2::Score`; the final
pick in `MkBuilder.cc`, `trackScoreV2p2Llh()`.

During the search each candidate carries a score that is the sum of one term
per layer step. The end-of-layer selection compares candidates by it, and the
final pick of one candidate per seed starts from it. V1 and V2 instead use
`track_score_func`, a formula evaluated on the whole track.

**Why additive.** With one term per layer, paths of different candidates are on
one scale after every layer, and the end-of-layer selection is a single flat
sort. `track_score_func` is not additive: with the default scorer the hit bonus
grows with the hit count, so the total is quadratic in it, and a hole is
counted as inside or tail depending on whether a hit follows later, so an old
hole changes its cost when a new hit is found.

**The form.** Each layer step scores the log-likelihood ratio of taking a hit
against taking none:

```
take hit j :  ln(eps/(1-eps)) - ln(2 pi) - chi2_j/2 - ln(rho sqrt(det V_j))
take none  :  0
```

eps is the hit efficiency of the layer (`hit_eff`), V the 2x2 covariance of the
residual and rho the density of hits in the candidate's search window, in hits
per cm^2. The hit term compares the Gaussian density of the residual,
exp(-chi2/2) / (2 pi sqrt(det V)), with the density rho of random hits; their
ratio is a pure number, so the units cancel. This is `Score::mode` 1, the
default.
A hit has to fit better than a random hit at that density would; a wrong hit in
a dense region therefore loses to a hole. A step with several hits adds one such
term per hit. A hole scores 0: within a layer the question is only whether to
take a hit. Outward and inward searches have separate parameter sets
(`V2p2Params::score_outward`, `score_inward`).

`Score::mode` 0 is the earlier linear form, a bonus per hit and per overlap hit
minus a chi2 weight and a penalty per hole, kept for comparison. It needs a hit
bonus small enough that a wrong hit loses to a hole, and with that it shortens
the tracks; the likelihood removes the same fakes without shortening them. The
measurements are in [History](#appendix-history).

**The hit efficiency is a tuned prior.** eps enters only through
ln(eps/(1-eps)), the prior odds of a hit, and so sets how well a hit must fit to
beat a hole. Its values come from tuning on CMSSW MTV (2026-10), not from the
detector: outward 0.10, with 0.14 in the OT PS barrel and 0.20 in the OT 2S
barrel (`hit_eff_grp` of the score sets in `V2p2Params`); inward 0.90. Outward,
0.10 asks for 4.4 more in log-likelihood per hit than 0.90 would. Why the
outward search needs this is not understood. The tuning is conditional on the
seeds, the duplicate cleaner and the final fit, and is to be redone when any of
them changes.

**Terms fixed at their mean** (`Score::use_rho`, `use_detv`). Replacing ln rho,
or ln det V, by its measured mean keeps the term's average and removes its
variation, which measures what the variation is worth. On F30 in mode 1 with
eps 0.99, removing the variation of rho costs 114 found tracks and removing that
of det V costs 67 (quality-val counts, at an earlier working point).

**The final pick.** At the end of the forward search each seed keeps one
candidate. Because holes score 0, a candidate with many holes is not penalised
for them during the search, and finished candidates of different length are not
yet comparable. The final pick (`v2p2:llh`, named per iteration in
`IterationConfig`) therefore takes the summed layer-step score and subtracts 8
per inside hole and 3 per tail hole (`Policy::final_pick_hole_penalty`,
`final_pick_tail_penalty`). At the end it is known which holes are which.

CMSSW MTV, `TrackBuilding/initialStep`, 100 events, CMSSW_20_1_0_pre2, the code
of this branch as of 2026-10-30 with either builder, on the same events; the
efficiency difference is 13 found sim tracks of about 6770:

| | efficiency | fake rate | duplicates | mean hits |
|---|---|---|---|---|
| V2 | 66.94 % | 1.92 % | 0.17 % | 14.30 |
| v2p2 | 67.13 % | 1.72 % | 0.13 % | 14.37 |

**Adding a term.** `LayerStepFeatures` carries what the score reads of a layer
step: hits taken, the chi2 sum, ln det V, the hit density, the hole kind, the
layer and the direction. A new term adds a member to it without changing a
function signature. The struct can be written to the trace, to fit a score
offline. A term that depends on the step, such as its length or its material,
should take the step from the candidate's last found hit
(`getLastFoundHitLyr()`), not from the previous plan entry: a candidate that
missed a layer steps across two.

## Backward fit

**Code:** `MkFinder::bkFitFitTracksV2p2()` in `MkFinderV2p2-BkFit.cc`,
`TrackerInfo::get_double_layers()`, `getS()` and `getSNearPlanePoint()` in
`PropagationMPlexPlane.cc`.

After the forward search, each track is fitted again from its last hit back to
its first. The fit propagates to each hit's module plane and updates there; it
needs no layer plan. Every hit is fitted, sister and overlap hits included,
where the V1 and V2 fit keeps only the first of consecutive hits in a layer. A
hit that is on the track twice, which merging the seed hits can cause, is
fitted once.

**Hit order within a physical layer.** The hits of a track are stored in the
order they were added: plan order for the two sensors of a module pair (see
[Reduction](#reduction-and-hit-ordering)), the seed's order for seed hits.
Neither is always the order in which the track meets them. So when the fit
reaches a physical layer, it takes all of the track's hits on that layer and
fits them nearest module plane first. The distance is measured along a straight
line, from the state at the start of the layer. Which two sensor layers make up
one physical layer comes from `LayerNumberConverter::doubleLayerPartner()`: the
mono and the stereo layer of each CMS layer or disk with stereo modules, the
stereo layer first in phase 2 and the mono layer first in phase 1. The geometry
builders, `MkFitGeometryESProducer` and the standalone's geometry plugins, pass it
to `TrackerInfo::setup_double_layers()`; the pairs are not stored in the geometry
file.

Measured on 20 events, against the order of the sim hits: before the change the
fit met the hits of a physical layer in the wrong order in 50.5 % of the
physical layers with two or more hits, after it in 0.17 %. The median d(pT)/pT of the fitted
tracks went from -0.0045 to -0.0018; found and fake tracks, with MTV's
association, did not change.

**The crossing near the turning point.** The propagation to a plane solves a
quadratic in the path length and takes the root of smaller size. Where a
low-pT track turns back in radius, the two roots are of similar size, and the
smaller one can be the track's other crossing of the same plane, 50 to 110 cm
from the hit. A fit that steps there loses the track: the median final pT of
such tracks was 0.18 of the true one. These steps were from 67 to 110 cm radius into the
barrel, at pT 0.4 to 0.7 GeV. So when the two roots are within a factor of
about 50 of each other, the propagation computes both crossing points and keeps
the one nearer the module. A step that starts on the plane, as from a sister
sensor, has one root near zero and the other far away, and is not flagged.
The rule is in the plane propagation, so it applies to every caller: the
backward fits of v2p2, V1 and V2, and the in-layer search.

Measured on 20 events. Tracks whose fitted state leaves the plane of a hit by
more than 0.1 cm went from 197 to 7. The rule took the other root in 202 steps
of the v2p2 backward fit, 326 of the V1 and V2 backward fit, and 20 of 657
thousand in the in-layer search; the forward search's output did not change.
Found and fake tracks with MTV's association did not change, and the time was
the same within the spread of repeated runs.

## V2 compatibility

**Code:** `MkFinder::selectHitIndicesV2()`.

V2 keeps its own window code, as in upstream CMSSW, and does not use the v2p2
window classes (`MkBins`), so the v2p2 window work does not change it. Its text
differs from upstream only where an interface changed: the propagation to a
module plane takes the module's position and normal, and three helpers of a
debug dump were renamed.

V2 does run on the new mini-propagators. The propagation to a radius is in
closed form, where upstream iterates a fixed number of times without a
convergence test, and it clamps the target radius when the layer cannot be
reached (see [Reachability](#reachability)). The two agree where the iteration
converges.

The upstream window has two properties that the v2p2 window does not: its phi
bin range stops before the bin holding the upper edge, which a margin of 2.75
half-bins makes up for, and its phi error is taken at the incoming state rather
than at the layer. Both are kept, so that V2's window stays upstream's. V2's
output is not identical to upstream, because of the new propagation.

## Open items

- The pre-selection cap should become per layer, since how often modules
  overlap varies strongly across the detector
  ([Reduction](#reduction-and-hit-ordering)).
- The reachability margin should scale with the track's expected dq, per layer,
  and be retuned on outward searches, pixel seeds and the full pT range
  ([Reachability](#reachability)).
- `Window::dphi_trk_fac` and `chi2_trk_fac` compensate a track covariance that
  is too small: much too small in the transition and endcap regions, attributed
  to the material description, and somewhat too small everywhere, not yet
  explained. Both should come down when the covariance is right ([Search
  window](#search-window), [Kalman hit acceptance](#kalman-hit-acceptance)).
- A fetch widened by the surface reference would remove the spare q bin
  ([The fetch](#the-fetch)).
- The endcap q half-length is a factor sqrt(2) too large for pixels
  ([Hit extents](#hit-extents)).
- A Hermite cubic from one state, the updated one, would be cheaper than the
  plane solve for the second and later hits of a path
  ([In-layer combinatorial search](#in-layer-combinatorial-search)).
- No phase-2 layer sets an r hole, so the gap verdict never fires
  ([Within-sensitive-region verdict](#within-sensitive-region-verdict)).
- The stop reason is not yet exported to CMSSW
  ([Candidate pickup and stopping cuts](#candidate-pickup-and-stopping-cuts)).
- The score's working point is conditional on the seeds, the duplicate cleaner
  and the final fit, and is to be redone when any of them changes
  ([Layer-step score](#layer-step-score)).

## Appendix: transport of the position covariance

**Code:** `MkBins::transport_position_cov()`.

The position block of the covariance is transported from the previous hit to
the layer's far crossing (`m_sp2`), at the path length s the propagation to the
layer reached it with. The track parameters are the position and (ipt, phi,
theta), with ipt = 1/pT. At fixed s the position is
x = x0 + f(ipt, phi, theta; s), so

```
C_pos(s) = J C0 J^T,   J = [ P_in | dx/d(ipt, phi, theta) ]   (3 x 6)
```

where J maps the six track parameters to the three position coordinates, with
the derivatives in closed form from quantities the propagation already holds.
With the helix in the turning angle a, inv_k the charge-signed field factor
(1/R = |ipt inv_k|; its sign gives the sense of rotation), k = 1/inv_k and p in
GeV,

```
x = x0 + k (px sin a - py (1 - cos a)),  y = y0 + k (py sin a + px (1 - cos a)),
z = z0 + s cos(theta),                   a = s sin(theta) ipt inv_k
```

so, with (dx, dy) the displacement and p_end the transverse momentum at `m_sp2`,

```
d(x,y)/dphi   = (-dy, dx)
d(x,y)/dipt   = (-(dx, dy) + a k p_end) / ipt = k (px f1 - py f2, py f1 + px f2) / ipt
d(x,y)/dtheta = a k p_end cot(theta),     dz/dtheta = -a k / ipt
f1 = a cos a - sin a,   f2 = a sin a - (1 - cos a)
```

with p at the previous hit in the second form of the ipt column. As in mkFit's
full propagation the result is curvilinear at both ends: `P_in` projects the
starting position onto the plane normal to the momentum there, and the result is
projected onto the plane normal to the momentum at `m_sp2`. The dq surface
reference in "Search window" depends on that. The first form of the ipt column
is a difference of two O(s) terms that nearly cancel for a stiff track: it lost
25 % in sigma_y at pT 137 GeV. The second form is used, with f1 and f2 from
their series below |a| = 0.25, where they are O(a^3) and O(a^2).

## Appendix: history

Superseded versions and measurements of earlier states of the code, kept for
reference. Times here are not comparable with the current ones.

- **Looper stop in V1** (see [Candidate pickup and stopping
  cuts](#candidate-pickup-and-stopping-cuts)). V1 tests |posPhi - momPhi|
  against 78.5 deg and, for angles that wrap around, against the literal 4.512
  rad. That is pi plus the limit, where 2 pi minus the limit was meant, so
  tracks between 78.5 and 101.5 deg pass whenever the two angles lie on opposite
  sides of the phi wrap.
- **Track covariance at the layer.** The transport replaced a full
  `propagateHelixToPlaneMPlex` call, of which only the six elements of the
  position block were read. On 20 events the quality-val track counts were
  identical in the default and two varied configurations, and the block went
  from 0.68 s to 0.23 s over those events in a trace build.
- **phi spare bin.** Before the line pre-cut one spare phi bin cost 20 % of the
  time (T100: 60.79 s against 48.38 s), since every fetched hit then cost a plane
  solve per candidate.
- **Path order.** Until 2026-10-01 the sign in the ordering key was the search
  direction alone, so for one charge every layer's hits were walked in reverse
  path order.
- **Best-hit path.** The first version of v2p2 ran one Kalman update per
  pre-selected hit, all from the candidate's incoming state, and kept the lowest
  chi2, so at most one hit per layer. It was removed once the in-layer search
  replaced it.
- **Beam width.** Before the line pre-cut, T100 with the in-layer search on was
  60.79 s at `maxCandsPerSeed` 3, the same time as with the spare phi bin above,
  so presumably the same configuration, and 76.45 s at 4.

### Layer-step score: from the hit bonus to the likelihood

**The hit bonus (mode 0).** In mode 0 a hit beats a hole when `hit_bonus -
chi2_weight * chi2 > -miss_penalty`, i.e. below chi2 = `(hit_bonus +
miss_penalty) / chi2_weight`. With the former bonus of 30 that was chi2 38, and
almost every hit that passed the acceptance cut beat a hole. With the bonus of
3 a hole wins above chi2 11. A low-pT candidate that ran into a neighbouring
particle's outer-tracker hits took them. In CMSSW MTV on ttbar PU200 D121, v2p2
had a fake rate of 4.67 % against 1.98 % for V2. The extra fakes were tracks of
6-16 hits with a correct pixel seed, and their outer-tracker hits belonged
mostly to another particle.

MTV, `TrackBuilding/initialStep`, 100 events, V2 = CMSSW_20_1_0_pre2 as
released, v2p2 = this branch as of 2026-09-28, before a fix of the layers'
inner radii in the geometry:

| | efficiency | fake rate | mean hits |
|---|---|---|---|
| V2 | 66.92 % | 1.98 % | 14.17 |
| v2p2, bonus 30 | 66.57 % | 4.67 % | 15.57 |
| v2p2, bonus 10 | 66.89 % | 3.27 % | 15.32 |
| v2p2, bonus 5 | 66.94 % | 2.79 % | 15.05 |
| v2p2, bonus 3 | 66.94 % | 2.57 % | 14.72 |

The remaining excess is at 1.7 < |eta| < 2.7 and pT < 0.5 GeV. Standalone, 50
events, MTV association: below a bonus of 3 the found and fake counts stop
improving, while tracks become shorter than V2's, the d(pT)/pT width grows and
duplicates rise.

**Mode 1 replaces mode 0.** The linear score cannot remove the forward fakes without
shortening the tracks. The log-likelihood can: a hit has to be better than the
local hit density predicts for a random one, so a wrong hit in a dense region
loses to a hole. With `hit_eff` 0.95 it added fakes at 0.9 < |eta| < 1.7; 0.90
does not. Same MTV setup; the mode 1 row also has the in-layer path order fix
and v2p2's own backward fit, the mode 0 row is before both:

| | efficiency | fake rate | duplicates | mean hits |
|---|---|---|---|---|
| V2 | 66.92 % | 1.98 % | 0.18 % | 14.17 |
| v2p2, mode 0, bonus 3 (a later run than the table above) | 66.94 % | 2.55 % | 0.12 % | 14.79 |
| v2p2, mode 1, hit_eff 0.90 | 67.03 % | 2.08 % | 0.15 % | 14.71 |

At that time the rest of the fake excess was in the barrel (1.15 % against
0.93 %) and at 0.9 < |eta| < 1.7 (2.87 % against 2.36 %). The current working
point is in [Layer-step score](#layer-step-score).
