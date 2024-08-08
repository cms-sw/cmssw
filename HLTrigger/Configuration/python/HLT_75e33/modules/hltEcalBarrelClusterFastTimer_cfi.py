import FWCore.ParameterSet.Config as cms

hltEcalBarrelClusterFastTimer = cms.EDProducer("EcalBarrelClusterFastTimer",
    ebClusters = cms.InputTag("hltParticleFlowClusterECALUncorrected"),
    ebTimeHits = cms.InputTag("hltEcalDetailedTimeRecHit","EcalRecHitsEB"),
    ecalDepth = cms.double(7.0),
    minEnergyToConsider = cms.double(0.0),
    minFractionToConsider = cms.double(0.1),
    resolutionModels = cms.VPSet(cms.PSet(
        modelName = cms.string('PerfectResolutionModel')
    )),
    timedVertices = cms.InputTag("hltOfflinePrimaryVertices4D")
)
