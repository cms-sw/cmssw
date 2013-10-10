import FWCore.ParameterSet.Config as cms

siStripSeeds = cms.EDProducer("SiStripElectronSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    SeedConfiguration = cms.PSet(
        beamSpot = cms.InputTag("offlineBeamSpot"),
        measurementTrackerName = cms.string(""),
        measurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
        tibOriginZCut = cms.double(20.),
        tidOriginZCut = cms.double(20.),
        tecOriginZCut = cms.double(20.),
        monoOriginZCut = cms.double(20.),
        tibDeltaPsiCut = cms.double(0.1),
        tidDeltaPsiCut = cms.double(0.1),
        tecDeltaPsiCut = cms.double(0.1),
        monoDeltaPsiCut = cms.double(0.1),
        tibPhiMissHit2Cut = cms.double(0.006),
        tidPhiMissHit2Cut = cms.double(0.006),
        tecPhiMissHit2Cut = cms.double(0.007),
        monoPhiMissHit2Cut = cms.double(0.02),
        tibZMissHit2Cut = cms.double(0.35),
        tidRMissHit2Cut = cms.double(0.3),
        tecRMissHit2Cut = cms.double(0.3),
        tidEtaUsage = cms.double(1.2),
        tidMaxHits = cms.int32(4),
        tecMaxHits = cms.int32(2),
        monoMaxHits = cms.int32(4),
        maxSeeds = cms.int32(5)
    )
    
)


