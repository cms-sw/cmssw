import FWCore.ParameterSet.Config as cms

CosmicMuonSeed = cms.EDFilter("CosmicMuonSeedGenerator",
    MaxSeeds = cms.int32(10),
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    EnableDTMeasurement = cms.bool(True),
    MaxCSCChi2 = cms.double(300.0),
    MaxDTChi2 = cms.double(300.0),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
    EnableCSCMeasurement = cms.bool(True)
)



