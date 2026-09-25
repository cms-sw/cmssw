#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Local-reco orchestration: attach HGCAL local reco upstream of TICL, validate
the boundary type-aware, and schedule the combined chain on a process."""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets, localreco
from RecoTICL.Configuration.model import PyTICLError
from RecoTICL.Configuration.validator import _product_graph, validate_assembled
from RecoTICL.Configuration.catalog import T_CALOCLUSTERS, T_MASK, T_TIME


def main():
    ok = True

    # build_full_reco validates the local-reco -> TICL boundary
    try:
        asm = localreco.build_full_reco(presets.v5())
        print("OK   : full reco (HGCAL local reco + TICL) validates")
    except PyTICLError as exc:
        print("FAIL : full reco did not validate:\n%s" % exc); return 1

    # the merge now provides exactly the products TICL's layer_clusters/
    # original_mask/time_layerclusters consume
    g = _product_graph(asm.modules)
    want = {
        ("hgcalMergeLayerClusters", ""): T_CALOCLUSTERS,
        ("hgcalMergeLayerClusters", "InitialLayerClustersMask"): T_MASK,
        ("hgcalMergeLayerClusters", "timeLayerCluster"): T_TIME,
    }
    for key, typ in want.items():
        if typ not in g.get(key, set()):
            print("FAIL : merge does not provide %s as %s" % (key, typ)); ok = False
    if ok:
        print("OK   : hgcalMergeLayerClusters feeds TICL (clusters + mask + time)")

    # boundary is really type-checked: a wrong-type layer-cluster source is rejected
    asm2 = presets.v5().assemble()
    asm2.modules["ticlTrackstersCLUE3DHigh"].layer_clusters = cms.InputTag("ticlCandidate")
    try:
        validate_assembled(asm2)
        print("FAIL : type-incompatible layer_clusters not rejected"); ok = False
    except PyTICLError:
        print("OK   : type-incompatible layer-cluster source is rejected")

    # the combined chain schedules local reco + TICL together
    p = cms.Process("TEST")
    full = localreco.add_to_process(p, localreco.build_full_reco(presets.v5(), validate=False))
    mods = full.moduleNames()
    if "hgcalMergeLayerClusters" not in mods or "ticlTrackstersCLUE3DHigh" not in mods:
        print("FAIL : combined fullRecoTask missing modules"); ok = False
    else:
        print("OK   : fullRecoTask schedules local reco + TICL (%d modules)" % len(mods))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
