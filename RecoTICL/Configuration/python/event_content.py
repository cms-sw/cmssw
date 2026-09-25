# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Event Content from the assembled pyTICL graph.

pyTICL assembles every TICL module and, through the :mod:`catalog`, knows the
products each one puts into the event.  That makes it the natural single source
of truth for the TICL output commands: instead of a hand-maintained list of
``keep`` statements that silently drifts when an iteration or a stage is added,
the keep-statements are derived from the configuration itself.

A :class:`~RecoTICL.Configuration.model.TICLConfig` declares which reconstruction
stages exist (iterations, links, superclustering, candidate, pf); the selected
:class:`~RecoTICL.Configuration.target.Target` maps each stage to its module
label.  This module turns that into the persisted output, by tier.  Intermediate
products (seeding regions, filtered-cluster masks, layer tiles, MTD SoA) are
transient and never kept.
"""

import FWCore.ParameterSet.Config as cms

# Output tiers, increasing in verbosity: AOD subset of RECO subset of FEVT.
AOD = "AOD"
RECO = "RECO"
FEVT = "FEVT"
_TIER_ORDER = (AOD, RECO, FEVT)

# Persistence policy: which assembled reconstruction stages are kept, and at
# which tier.  Anything the graph produces that is not listed here is transient.
_PERSIST = {
    "trackster": RECO,      # the per-iteration tracksters
    "links": RECO,          # ticlTracksterLinks (hadronic linking)
    "supercluster": RECO,   # ticlTracksterLinksSuperclustering* (EGamma)
    "candidate": RECO,      # ticlCandidate
    "pf": RECO,             # pfTICL
}


def _tiers_up_to(tier):
    if tier not in _TIER_ORDER:
        raise ValueError("unknown event-content tier %r (known: %s)"
                         % (tier, ", ".join(_TIER_ORDER)))
    return set(_TIER_ORDER[:_TIER_ORDER.index(tier) + 1])


def persisted_labels(cfg, tier=RECO):
    """Module labels persisted up to and including ``tier``, in assembly order.

    Derived from the config's declared stages and the target's label scheme, so
    the offline and HLT targets yield their own (``ticl*`` / ``hltTicl*``) labels
    from the same declaration.
    """
    target = cfg.target
    wanted = _tiers_up_to(tier)
    out, seen = [], set()

    def take(role, label):
        if label and _PERSIST.get(role) in wanted and label not in seen:
            seen.add(label)
            out.append(label)

    for it in cfg.iterations:
        if getattr(it, "persist", True):
            take("trackster", target.trackster_label(it.name))
    if cfg.links_spec:
        take("links", target.links_label)
    if cfg.superclustering_spec:
        take("supercluster", target.supercluster_dnn_label)
    if cfg.include_candidate:
        take("candidate", target.candidate_label)
    if cfg.include_pf:
        take("pf", target.pf_label)
    return out


def keep_statements(cfg, tier=RECO):
    """``keep *_<label>_*_*`` for every module persisted up to ``tier``."""
    return ["keep *_%s_*_*" % label for label in persisted_labels(cfg, tier)]


def output_commands(cfg, tier=RECO):
    """The tier's keep-statements as a ``cms.untracked.vstring``."""
    return cms.untracked.vstring(*keep_statements(cfg, tier))


def event_content_pset(cfg, tier=RECO):
    """A ``cms.PSet(outputCommands=...)`` for the tier (an EventContent block)."""
    return cms.PSet(outputCommands=output_commands(cfg, tier))
