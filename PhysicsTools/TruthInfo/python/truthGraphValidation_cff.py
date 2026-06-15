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
    simHitToRecHitMapProducer,
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

# Producers (truth graph + hit index + association maps) followed by the DQM
# analyzers. Append to a validation sequence with the calo truth available.
truthGraphValidationSequence = cms.Sequence(
    truthGraphProducer
    + truthLogicalGraphProducer
    + simHitToRecHitMapProducer
    + truthLogicalGraphHitIndexProducer
    + truthBranchCaloAssociationProducer
    + branchHGCalValidator
)
