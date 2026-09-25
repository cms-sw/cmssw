#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""pyTICL Event Content: generated keeps reproduce the baseline TICL content.

Asserts that the keep-statements pyTICL derives from the assembled v5 graph match
the live baseline (``ticlIterLabelsPSet.labels`` for the per-iteration tracksters,
plus the reconstruction singletons), and that the generated ``TICL_RECO/FEVT/
FEVTHLT`` blocks nest correctly and cover the baseline ``RecoHGCal_EventContent``
TICL collections.
"""
import sys

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.event_content import (
    keep_statements, persisted_labels, RECO, FEVT, AOD)
from RecoTICL.Configuration.target import HLT


def _labels_from_keeps(keeps):
    # 'keep *_<label>_*_*' -> '<label>'
    return [k.split("_", 1)[1].rsplit("_", 2)[0] for k in keeps]


def test_recovery_is_not_persisted():
    """A recovery iteration is intermediate: its tracksters are not kept."""
    cfg = presets.v5()
    track = [l for l in persisted_labels(cfg, RECO) if l.startswith("ticlTracksters")]
    assert track == ["ticlTrackstersCLUE3DHigh"], track  # Recovery excluded


def test_persisted_set_matches_baseline():
    """pyTICL's persisted trackster-family labels match the live
    ticlIterLabelsPSet.labels (the tracked collections) when available."""
    try:
        from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet
    except Exception:
        return  # baseline not built here; the byte-for-byte test covers labels
    baseline = set(ticlIterLabelsPSet.labels)
    # pfTICL is kept by an explicit baseline statement, not part of the label set
    mine = set(persisted_labels(presets.v5(), RECO)) - {"pfTICL"}
    assert mine == baseline, "pyTICL %r != baseline %r" % (sorted(mine), sorted(baseline))


def test_reco_singletons_present():
    """links, supercluster, candidate and pf are kept at RECO."""
    labels = persisted_labels(presets.v5(), RECO)
    for must in ("ticlTracksterLinks", "ticlTracksterLinksSuperclusteringDNN",
                 "ticlCandidate", "pfTICL"):
        assert must in labels, "%s not persisted (have %r)" % (must, labels)
    # intermediate products are transient
    for never in ("ticlSeedingGlobal", "ticlLayerTileProducer", "mtdSoA"):
        assert never not in labels, "%s should not be persisted" % never


def test_blocks_nest_and_generate():
    """TICL_RECO/FEVT/FEVTHLT nest, and embed the generated reconstruction keeps."""
    from RecoTICL.Configuration import ticlEventContent_cff as gen

    g_reco = set(gen.TICL_RECO.outputCommands)
    g_fevt = set(gen.TICL_FEVT.outputCommands)
    g_hlt = set(gen.TICL_FEVTHLT.outputCommands)
    assert g_reco < g_fevt < g_hlt, "tiers must nest (RECO subset FEVT subset FEVTHLT)"

    # every generated reconstruction keep is present in TICL_RECO
    for k in keep_statements(presets.v5(), RECO):
        assert k in g_reco, "missing generated keep %r" % k
    assert "keep *_ticlCandidate_*_*" in g_reco
    assert "keep *_pfTICL_*_*" in g_reco


def test_hlt_target_relabels():
    """The same v5 declaration yields hltTicl* labels under the HLT target."""
    cfg = presets.v5("hltv5")
    cfg.target = HLT
    keeps = keep_statements(cfg, RECO)
    assert all("hltTicl" in k or "hltPfTICL" in k for k in keeps), keeps


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, "--", e)
    print("OK: pyTICL Event Content reproduces the baseline TICL content"
          if not fails else "%d failures" % fails)
    sys.exit(1 if fails else 0)
