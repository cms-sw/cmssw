import FWCore.ParameterSet.Config as cms

ecalBarrelClusterFastTimer = cms.EDProducer("EcalBarrelClusterFastTimer",
    ebClusters = cms.InputTag("particleFlowClusterECALUncorrected"),
    ebTimeHits = cms.InputTag("ecalDetailedTimeRecHit","EcalRecHitsEB"),
    ecalDepth = cms.double(7.0),
    minEnergyToConsider = cms.double(0.0),
    minFractionToConsider = cms.double(0.1),
    resolutionModels = cms.VPSet(cms.PSet(
        modelName = cms.string('PerfectResolutionModel')
    )),
    timedVertices = cms.InputTag("offlinePrimaryVertices4D")
)
