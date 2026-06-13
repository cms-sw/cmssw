#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

"""Diagnose connectivity of the SimTrack/SimVertex graph in a SIM/RECO file.

The truth graph is fragmented whenever a stored secondary SimTrack has its
production SimVertex parentIndex == -1, which happens if the Geant4 mother was
not persisted (default PersistencyEmin = 50 GeV drops soft intermediate
ancestors). With the enableTruth workflows (PersistencyEmin = 0) every stored
track keeps its full ancestor branch, so every component traces back to a
generator primary and orphan_components -> 0.

This reads the g4SimHits SimTrackContainer/SimVertexContainer via FWLite and,
per event, reports the number of weakly-connected components and how many are
NOT connected to a generator primary (the orphans).

Requires cmsenv. Examples:
  cmsenv
  truthGraphConnectivity.py step3.root
  truthGraphConnectivity.py step1.root -n 5
  truthGraphConnectivity.py step3.root --link ancestor   # use SimTrack ancestor id
  truthGraphConnectivity.py step3.root --link combined    # parentIndex, then ancestor
"""
import argparse
import sys

from DataFormats.FWLite import Events, Handle


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def analyze_event(tracks, verts, link):
    """Return a dict of connectivity metrics for one event.

    link selects how a non-primary track is connected to its parent:
      'parentIndex' - via its production SimVertex.parentIndex (default; this is
                      the link the truth-graph producer uses),
      'ancestor'    - via SimTrack.getPrimaryOrLastStoredID(),
      'combined'    - parentIndex first, ancestor as fallback.
    """
    tid_to_i = {t.trackId(): i for i, t in enumerate(tracks)}
    stored = set(tid_to_i)
    nT, nV = len(tracks), len(verts)
    parent = list(range(nT))

    def is_primary(t):
        return t.isPrimary() and t.genpartIndex() != -1

    orphan_vtx = 0  # vertices whose parentIndex points to an unsaved track
    for v in verts:
        pi = v.parentIndex()
        if pi > 0 and pi not in stored:
            orphan_vtx += 1

    for i, t in enumerate(tracks):
        if is_primary(t):
            continue
        linked = False
        if link in ('parentIndex', 'combined'):
            vi = t.vertIndex()
            pi = verts[vi].parentIndex() if 0 <= vi < nV else -1
            if pi > 0 and pi in stored:
                _union(parent, i, tid_to_i[pi])
                linked = True
        if not linked and link in ('ancestor', 'combined'):
            anc = t.getPrimaryOrLastStoredID()
            if anc > 0 and anc in stored:
                _union(parent, i, tid_to_i[anc])

    gen_roots = set(_find(parent, i) for i, t in enumerate(tracks) if is_primary(t))
    comps = set(_find(parent, i) for i in range(nT))
    comps_with_gen = sum(1 for c in comps if c in gen_roots)
    orphan_tracks = sum(1 for i in range(nT) if _find(parent, i) not in gen_roots)

    return dict(nTrk=nT, nVtx=nV, primaries=len(gen_roots), orphan_vertices=orphan_vtx,
                components=len(comps), components_with_gen=comps_with_gen,
                orphan_components=len(comps) - comps_with_gen,
                orphan_tracks=orphan_tracks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputFile", help="SIM or RECO root file (must contain g4SimHits SimTracks/SimVertices)")
    ap.add_argument("-n", "--maxEvents", type=int, default=3, help="events to process, -1 for all (default 3)")
    ap.add_argument("-l", "--label", default="g4SimHits", help="SimTrack/SimVertex module label (default g4SimHits)")
    ap.add_argument("--link", choices=("parentIndex", "ancestor", "combined"), default="parentIndex",
                    help="how to connect a secondary to its parent (default parentIndex)")
    args = ap.parse_args()

    src = args.inputFile if (":" in args.inputFile or args.inputFile.startswith("/")) else "file:" + args.inputFile
    events = Events(src)
    hT, hV = Handle("std::vector<SimTrack>"), Handle("std::vector<SimVertex>")

    any_orphan = False
    for iev, ev in enumerate(events):
        if args.maxEvents >= 0 and iev >= args.maxEvents:
            break
        ev.getByLabel(args.label, hT)
        ev.getByLabel(args.label, hV)
        m = analyze_event(list(hT.product()), list(hV.product()), args.link)
        any_orphan = any_orphan or m["orphan_components"] > 0
        print(f"event {iev}: nTrk={m['nTrk']} nVtx={m['nVtx']} primaries={m['primaries']} "
              f"orphanVtx(parentUnsaved)={m['orphan_vertices']}")
        print(f"  components={m['components']} with_gen={m['components_with_gen']} "
              f"orphan_components={m['orphan_components']} "
              f"orphan_tracks={m['orphan_tracks']}/{m['nTrk']} "
              f"({100.0 * m['orphan_tracks'] / max(m['nTrk'], 1):.1f}%)")

    # non-zero exit if any event had disconnected components (handy in CI/checks)
    sys.exit(1 if any_orphan else 0)


if __name__ == "__main__":
    main()
