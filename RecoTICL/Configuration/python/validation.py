# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Single-source-of-truth derivation of the TICL labels, associators, dumper and
validator.

From one list of trackster-collection *labels*, pyTICL derives -- exactly as the
baseline ``iterativeTICL_cff`` / ``ticlDumper_cff`` / ``HGCalValidator_cff`` do --
the ``ticlIterLabelsPSet``, the ``associatorsInstances``, the
trackster<->simTrackster associator producers (``...ByLCs`` / ``...ByHits``,
auto-created **and** scheduled in a Task), the ``ticlDumper`` and the
``hgcalValidator``.  Keeping a single source of truth means reco, dumper and
validation can never drift out of sync.
"""

import FWCore.ParameterSet.Config as cms

SIM_COLLECTIONS = ["ticlSimTracksters", "ticlSimTrackstersfromCPs"]


# --------------------------------------------------------------------------- #
# label derivation
# --------------------------------------------------------------------------- #

def default_validation_labels(cfg):
    """The collections worth validating/dumping, derived from the config:
    the primary (first) iteration trackster + links + candidate + superclustering.
    For the v5 default this is exactly ``ticlIterLabelsPSet.labels``."""
    t = cfg.target
    labels = []
    if cfg.iterations:
        labels.append(t.trackster_label(cfg.iterations[0].name))
    if cfg.links_spec:
        labels.append(t.links_label)
    if cfg.include_candidate:
        labels.append(t.candidate_label)
    if cfg.superclustering_spec:
        labels.append(t.supercluster_dnn_label)
    return labels


def iter_labels_pset(labels):
    return cms.PSet(labels=cms.vstring(*labels))


def associator_instances(labels):
    """``label x simCollection x {RecoToSim, SimToReco}`` -- as in iterativeTICL_cff."""
    out = []
    for lab in labels:
        for sts in SIM_COLLECTIONS:
            out.append(lab + "To" + sts)
            out.append(sts + "To" + lab)
    return out


# --------------------------------------------------------------------------- #
# associator producers (auto-created)
# --------------------------------------------------------------------------- #

def build_associators_by_lcs(labels):
    from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByLCsProducer_cfi \
        import AllTracksterToSimTracksterAssociatorsByLCsProducer as base
    return base.clone(
        tracksterCollections=cms.VInputTag(*[cms.InputTag(l) for l in labels]),
        simTracksterCollections=cms.VInputTag(
            cms.InputTag("ticlSimTracksters"),
            cms.InputTag("ticlSimTracksters", "fromCPs"),
        ),
    )


def build_associators_by_hits(labels):
    from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi \
        import AllTracksterToSimTracksterAssociatorsByHitsProducer as base
    return base.clone(
        tracksterCollections=cms.VInputTag(*[cms.InputTag(l) for l in labels]),
        simTracksterCollections=cms.VInputTag("ticlSimTracksters", "ticlSimTracksters:fromCPs"),
    )


# --------------------------------------------------------------------------- #
# dumper & validator
# --------------------------------------------------------------------------- #

def build_ticl_dumper(labels):
    from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper as base
    dumper_associators = []
    for sts in SIM_COLLECTIONS:
        for lab in labels:
            suffix = "CP" if "fromCPs" in sts else "SC"
            dumper_associators.append(cms.PSet(
                branchName=cms.string(lab),
                suffix=cms.string(suffix),
                associatorRecoToSimInputTag=cms.InputTag(
                    "allTrackstersToSimTrackstersAssociationsByLCs:%sTo%s" % (lab, sts)),
                associatorSimToRecoInputTag=cms.InputTag(
                    "allTrackstersToSimTrackstersAssociationsByLCs:%sTo%s" % (sts, lab)),
            ))
    return base.clone(
        tracksterCollections=[
            *[cms.PSet(treeName=cms.string(l), inputTag=cms.InputTag(l)) for l in labels],
            cms.PSet(treeName=cms.string("simtrackstersSC"),
                     inputTag=cms.InputTag("ticlSimTracksters"),
                     tracksterType=cms.string("SimTracksterSC")),
            cms.PSet(treeName=cms.string("simtrackstersCP"),
                     inputTag=cms.InputTag("ticlSimTracksters", "fromCPs"),
                     tracksterType=cms.string("SimTracksterCP")),
        ],
        associators=dumper_associators,
        saveSuperclustering=cms.bool(True),
    )


def build_hgcal_validator(labels, primary_trackster="ticlTrackstersCLUE3DHigh",
                          merge_label="ticlCandidate"):
    from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as base
    inst = associator_instances(labels)
    return base.clone(
        label_tst=cms.VInputTag(
            *[cms.InputTag(l) for l in labels]
            + [cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")]),
        allTracksterTracksterAssociatorsLabels=cms.VInputTag(
            *[cms.InputTag("allTrackstersToSimTrackstersAssociationsByLCs:" + a) for a in inst]),
        allTracksterTracksterByHitsAssociatorsLabels=cms.VInputTag(
            *[cms.InputTag("allTrackstersToSimTrackstersAssociationsByHits:" + a) for a in inst]),
        LayerClustersInputMask=cms.VInputTag(
            cms.InputTag(primary_trackster),
            cms.InputTag("ticlSimTracksters", "fromCPs"),
            cms.InputTag("ticlSimTracksters")),
        ticlTrackstersMerge=cms.InputTag(merge_label),
        mergeSimToRecoAssociator=cms.InputTag(
            "allTrackstersToSimTrackstersAssociationsByLCs:ticlSimTrackstersfromCPsTo" + merge_label),
        mergeRecoToSimAssociator=cms.InputTag(
            "allTrackstersToSimTrackstersAssociationsByLCs:" + merge_label + "ToticlSimTrackstersfromCPs"),
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def build_validation(cfg):
    """Build all label-derived validation modules + the associator Task.

    Returns ``(modules, task_children, labels)`` where ``modules`` maps label->module
    and the associators are scheduled together in ``ticlAssociatorsTask``."""
    spec = cfg.validation_spec or {}
    labels = spec.get("labels") or default_validation_labels(cfg)
    t = cfg.target

    modules = {
        "allTrackstersToSimTrackstersAssociationsByLCs": build_associators_by_lcs(labels),
        "allTrackstersToSimTrackstersAssociationsByHits": build_associators_by_hits(labels),
    }
    primary = t.trackster_label(cfg.iterations[0].name) if cfg.iterations else "ticlTrackstersCLUE3DHigh"
    modules["hgcalValidator"] = build_hgcal_validator(labels, primary, t.candidate_label)
    if spec.get("dumper"):
        modules["ticlDumper"] = build_ticl_dumper(labels)

    task_children = {
        "ticlAssociatorsTask": [
            "allTrackstersToSimTrackstersAssociationsByLCs",
            "allTrackstersToSimTrackstersAssociationsByHits",
        ],
    }
    return modules, task_children, labels
