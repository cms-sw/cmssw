import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterCleaners_cfi \
     import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterSeeders_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterizers_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterEnergyCorrectors_cfi import *

particleFlowClusterECALUncorrectedWithTime = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitECALWithTime"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_ECAL),
    seedFinder = localMaxSeeds_ECAL,
    initialClusteringStep = topoClusterizer_ECAL,
    pfClusterBuilder = pfClusterizerWithTime_ECAL,
    positionReCalc = positionCalcECAL_all_withdepth
    )

particleFlowClusterECAL = cms.EDProducer(
    "CorrectedECALPFClusterProducer",
    inputECAL = cms.InputTag("particleFlowClusterECALUncorrectedWithTime"),
    inputPS = cms.InputTag("particleFlowClusterPS"),
    minimumPSEnergy = cms.double(0.0),
    energyCorrector = emEnergyCorrector
    )

