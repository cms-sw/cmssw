import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterCleaners_cfi \
     import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterSeeders_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterizers_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterEnergyCorrectors_cfi import *

particleFlowClusterECALBarrel = cms.EDProducer(
    "PFClusterProducerNew",
    recHitsSource = cms.InputTag("particleFlowRecHitECAL:EB"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_EB),
    seedFinder = localMaxSeeds_EB,
    topoClusterBuilder = topoClusterizer_EB,
    pfClusterBuilder = pfClusterizer_EB,
    positionReCalc = positionCalcECAL_all_withdepth
    )

particleFlowClusterECALEndcap = cms.EDProducer(
    "PFClusterProducerNew",
    recHitsSource = cms.InputTag("particleFlowRecHitECAL:EE"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_EE),
    seedFinder = localMaxSeeds_EE,
    topoClusterBuilder = topoClusterizer_EE,
    pfClusterBuilder = pfClusterizer_EE,
    positionReCalc = positionCalcECAL_all_withdepth
    )

particleFlowClusterECALUncorrected = cms.EDProducer(
    "PFClusterCollectionMerger",
    inputs = cms.VInputTag(
       cms.InputTag("particleFlowClusterECALBarrel"),
       cms.InputTag("particleFlowClusterECALEndcap"))
    )

particleFlowClusterECALNew = cms.EDProducer(
    "CorrectedECALPFClusterProducer",
    inputECAL = cms.InputTag("particleFlowClusterECALUncorrected"),
    inputPS = cms.InputTag("particleFlowClusterPS"),
    minimumPSEnergy = cms.double(0.0),
    energyCorrector = emEnergyCorrector
    )

