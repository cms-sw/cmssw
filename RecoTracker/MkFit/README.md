# mkFit

This package holds the glue modules for running
[mkFit](http://trackreco.github.io/) within CMSSW.

Note that at the moment there may be only one `MkFitProducer` in a
single job. This restriction will be removed in the future.

Also note that at the moment the mkFit works only with the CMS phase1
tracker detector. Support for the phase2 tracker will be added later.

## Modifier for runTheMatrix workflows (offline reconstruction)

* `Configuration.ProcessModifiers.trackingMkFit_cff.trackingMkFit`
  * Replaces initialStep track building module with `mkFit`.

## Customize functions for runTheMatrix workflows (offline reconstruction)

* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnly`
  * Run only the initialStep tracking. In practice this configuration
    runs the initialStepPreSplitting iteration, but named as
    initialStep. MultiTrackValidator is included, and configured to
    monitor initialStep. Intended to provide the minimal configuration
    for CMSSW tests.
* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnlyNoMTV`
  * Otherwise same as `customizeInitialStepOnly` except drops
    MultiTrackValidator. Intended for profiling.


These can be used with e.g.
```bash
$ runTheMatrix.py -l <workflow(s)> --apply 2 --command "--procModifiers trackingMkFit --customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepOnly"
```

## Description of configuration parameters

### Iteration configuration [class IterationConfig]

* *m_track_algorithm:* CMSSW track algorithm (used internally for reporting and consistency checks)
* *m_requires_seed_hit_sorting:* do hits on seed tracks need to be sorted (required for seeds that include strip layers)
* *m_require_quality_filter:* is additional post-processing required for result tracks
* *m_params:* IterationParams structure for this iteration
* *m_layer_configs:* std::vector of per-layer parameters

### Iteration parameters [class IterationParams]

* *nlayers_per_seed:* internal mkFit parameter used for standalone validation
* *maxCandsPerSeed:* maximum number of concurrent track candidates per given seed
* *maxHolesPerCand:* maximum number of allowed holes on a candidate
* *maxConsecHoles:*  maximum number of allowed consecutive holes on a candidate
* *chi2Cut:*         chi2 cut for accepting a new hit
* *chi2CutOverlap:*  chi2 cut for accepting an overlap hit
* *pTCutOverlap:*    pT cut below which the overlap hits are not picked up

#### Seed cleaning params (based on elliptical dR-dz cut)

* *c_ptthr_hpt:* pT threshold used to tighten seed cleaning requirements
* *c_drmax_bh:* dR cut used for seed tracks with std::fabs(eta)<0.9 and pT > c_ptthr_hpt
* *c_dzmax_bh:* dz cut used for seed tracks with std::fabs(eta)<0.9 and pT > c_ptthr_hpt
* *c_drmax_eh:* dR cut used for seed tracks with std::fabs(eta)>0.9 and pT > c_ptthr_hpt
* *c_dzmax_eh:* dz cut used for seed tracks with std::fabs(eta)>0.9 and pT > c_ptthr_hpt
* *c_drmax_bl:* dR cut used for seed tracks with std::fabs(eta)<0.9 and pT < c_ptthr_hpt
* *c_dzmax_bl:* dz cut used for seed tracks with std::fabs(eta)<0.9 and pT < c_ptthr_hpt
* *c_drmax_el:* dR cut used for seed tracks with std::fabs(eta)>0.9 and pT < c_ptthr_hpt
* *c_dzmax_el:* dz cut used for seed tracks with std::fabs(eta)>0.9 and pT < c_ptthr_hpt

#### Duplicate cleaning parameters

* *minHitsQF:* min number of hits on track candidate to apply duplicate cleaning based on fraction of shared hits
* *fracSharedHits:* min fraction of shared hits to determine duplicate track candidate

### Per-layer parameters [class IterationLayerConfig]

* *m_select_min_dphi, m_select_max_dphi:* geometry-driven dphi baseline selection window cut
* *m_select_min_dq, m_select_max_dq:* geometry-driven dr (endcap) / dz (barrel) baseline selection window cut
* *c_dp_[012]:* dphi selection window cut (= [0]*1/pT + [1]*std::fabs(theta-pi/2) + [2])
* *c_dp_sf:* additional scaling factor for dphi cut
* *c_dq_[012]:* dr (endcap) / dz (barrel) selection window cut (= [0]*1/pT + [1]*std::fabs(theta-pi/2) + [2])
* *c_dq_sf:* additional scaling factor for dr (endcap) / dz (barrel) cut
