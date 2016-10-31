import FWCore.ParameterSet.Config as cms

ecalBarrelClusterFastTimer = cms.EDProducer(
    'EcalBarrelClusterFastTimer',
    ebTimeHits = cms.InputTag('ecalDetailedTimeRecHit:EcalRecHitsEB'),
    ebClusters = cms.InputTag('particleFlowClusterECALUncorrected'),
    timedVertices = cms.InputTag('offlinePrimaryVertices4D'),
    minFractionToConsider = cms.double(0.1),
    minEnergyToConsider = cms.double(0.0),
    ecalDepth = cms.double(7.0),
    resolutionModels = cms.VPSet( cms.PSet( modelName = cms.string('PerfectResolutionModel') ) )
    )
