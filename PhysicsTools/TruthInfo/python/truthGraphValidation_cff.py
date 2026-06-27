# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Branch performance-plot validation: the truth-graph producers, the Branch<->reco
# association maps, and the DQM analyzers that turn them into plots comparing the
# truth::Branch graph to the legacy truth objects. Harvesting (efficiency) lives in
# truthGraphDQMHarvester_cff. Hooked into globalValidation behind enableTruth.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# Reuse the producer chain already defined for the prevalidation.
from Validation.Configuration.truthPrevalidation_cff import (
    truthGraphProducer,
    truthLogicalGraphProducer,
    detIdToRecHitMapProducer,
    truthLogicalGraphHitIndexProducer,
)

# TICL-style Branch <-> calo-truth association maps (best-matched branch first),
# restricted to the interesting particles via interestingPdgIds (empty = all).
truthBranchCaloAssociationProducer = cms.EDProducer(
    "TruthBranchCaloAssociationProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
    simClusters=cms.InputTag("mix", "MergedCaloTruth"),
    interestingPdgIds=cms.vint32(),
)

branchHGCalValidator = DQMEDAnalyzer(
    "BranchHGCalValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
    simClusters=cms.InputTag("mix", "MergedCaloTruth"),
    folder=cms.string("HGCAL/BranchValidator"),
    minPt=cms.double(1.0),
    maxEta=cms.double(3.0),
)

# Tracker counterpart. A TrackingParticle has no hits of its own, so the
# Branch<->TrackingParticle comparison is mediated by the reco track: the
# association producer matches reco tracks to branches by shared tracker simhits,
# and the validator closes the loop to the TrackingParticle via ClusterTPAssociation.
# Phase-2 tracker: pixel + outer-tracker (Phase2TrackerCluster1D), no strips.
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer as _tpClusterProducer
truthTpClusterProducer = _tpClusterProducer.clone(
    pixelClusterSrc=cms.InputTag("siPixelClusters"),
    phase2OTClusterSrc=cms.InputTag("siPhase2Clusters"),
    pixelSimLinkSrc=cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc=cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc=cms.InputTag("mix", "MergedTrackTruth"),
    throwOnMissingCollections=cms.bool(False),
)

truthBranchTrackingAssociationProducer = cms.EDProducer(
    "TruthBranchTrackingAssociationProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    interestingPdgIds=cms.vint32(),
)

branchTrackingValidator = DQMEDAnalyzer(
    "BranchTrackingValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    clusterTPMap=cms.InputTag("truthTpClusterProducer"),
    folder=cms.string("Tracking/BranchValidator"),
    minPt=cms.double(0.9),
    maxEta=cms.double(3.0),
)

# Generic reco-side validators: match a reco collection to the Branch graph by
# shared hits and book MTV/HGCalValidator-style efficiency/fake/merge/duplicate.
# One template, two instantiations driven by the truth::recoHits adapters.
#
# IMPORTANT (why these are EXPERIMENTAL/opt-in, see the sequence below): the sim
# reference (interestingPdgIds) must be a DISJOINT (antichain) set of particles. A
# Branch subgraph aggregates a particle's descendants, so against the full graph
# every ancestor contains its descendants' hits and every reco object "merges" >=2
# nested branches (merge-rate ~1, efficiency ~0). A flat PDG-id list is a sufficient
# antichain ONLY for non-showering species (e.g. muons); for showering species it is
# still degenerate. The physically correct reference is the BranchSelector
# "interesting particles" antichain (CaloParticle-like for calo, TrackingParticle-
# like for tracking) - detector-dependent and not yet wired. The values below are
# placeholders for that opt-in configuration; muons are the one species that already
# gives meaningful numbers (see test/validateBranchRecoDQM_cfg.py).
branchTrackRecoValidator = DQMEDAnalyzer(
    "BranchTrackRecoValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    recoCollection=cms.InputTag("generalTracks"),
    interestingPdgIds=cms.vint32(13, -13),  # muons: a near-antichain (do not shower)
    folder=cms.string("Tracking/BranchValidator/recoTrack"),
    xName=cms.string("pt"),
    xTitle=cms.string("p_{T} [GeV]"),
    xMax=cms.double(200.0),
    minX=cms.double(0.9),
    minAbsEta=cms.double(0.0),
    maxAbsEta=cms.double(3.0),
    matchThreshold=cms.double(0.5),
    mergeThreshold=cms.double(0.3),
)

branchTracksterRecoValidator = DQMEDAnalyzer(
    "BranchTracksterRecoValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    recoCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    # Placeholder; needs the BranchSelector calo antichain to be non-degenerate.
    interestingPdgIds=cms.vint32(22, 11, -11, 211, -211, 321, -321, 2212, -2212, 2112),
    folder=cms.string("HGCAL/BranchValidator/Trackster"),
    xName=cms.string("energy"),
    xTitle=cms.string("E [GeV]"),
    xMax=cms.double(500.0),
    minX=cms.double(1.0),
    minAbsEta=cms.double(1.5),
    maxAbsEta=cms.double(3.0),
    matchThreshold=cms.double(0.5),
    mergeThreshold=cms.double(0.3),
)

# Producers (truth graph + hit index + association maps) followed by the DQM
# analyzers that reproduce the legacy truth objects (CaloParticle/SimCluster via
# branchHGCalValidator, TrackingParticle via branchTrackingValidator - both verified
# meaningful). Append to a validation sequence with the calo truth, the reco tracks
# and the tracker digi sim-links available.
truthGraphValidationSequence = cms.Sequence(
    truthGraphProducer
    + truthLogicalGraphProducer
    + detIdToRecHitMapProducer
    + truthLogicalGraphHitIndexProducer
    + truthBranchCaloAssociationProducer
    + truthTpClusterProducer
    + truthBranchTrackingAssociationProducer
    + branchHGCalValidator
    + branchTrackingValidator
)

# Split views for wiring into the release validation: the EDProducers (truth graph,
# hit index, association maps) run in the prevalidation Path, the DQM analyzers in
# the validation EndPath. truthGraphValidationSequence (above) keeps both together
# for the standalone single-file drivers in test/.
truthGraphValidationProducers = cms.Sequence(
    truthGraphProducer
    + truthLogicalGraphProducer
    + detIdToRecHitMapProducer
    + truthLogicalGraphHitIndexProducer
    + truthBranchCaloAssociationProducer
    + truthTpClusterProducer
    + truthBranchTrackingAssociationProducer
)
truthGraphValidationAnalyzers = cms.Sequence(
    branchHGCalValidator
    + branchTrackingValidator
)

# EXPERIMENTAL, opt-in (NOT in the default sequence): the generic reco-side
# eff/fake/merge/duplicate validators are only meaningful against a DISJOINT
# (antichain) interesting-particle reference. Because a Branch subgraph aggregates
# descendants, against the full graph every ancestor contains its descendants' hits
# and every reco object merges nested branches (merge-rate ~1, efficiency ~0). A
# flat PDG-id list is NOT a sufficient antichain for showering species (it only
# works for non-showering ones such as muons). The correct reference is the
# BranchSelector "interesting particles" antichain (CaloParticle-like for calo,
# TrackingParticle-like for tracking), which is detector-dependent and not yet
# wired - so these run only on demand (see test/validateBranchRecoDQM_cfg.py).
truthGraphRecoSideValidationSequence = cms.Sequence(
    branchTrackRecoValidator + branchTracksterRecoValidator
)
