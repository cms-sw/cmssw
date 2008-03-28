import FWCore.ParameterSet.Config as cms

CosmicMuonSeed = cms.EDFilter("CosmicMuonSeedGenerator",
    MaxSeeds = cms.int32(10),
    CSCRecSegmentLabel = cms.untracked.InputTag("cscSegments"),
    EnableDTMeasurement = cms.untracked.bool(True),
    MaxCSCChi2 = cms.double(300.0),
    MaxDTChi2 = cms.double(300.0),
    DTRecSegmentLabel = cms.untracked.InputTag("dt4DSegments"),
    EnableCSCMeasurement = cms.untracked.bool(True)
)


