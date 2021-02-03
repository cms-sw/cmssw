import FWCore.ParameterSet.Config as cms

displacedMuonSeeds = cms.EDProducer("CosmicMuonSeedGenerator",
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
    EnableCSCMeasurement = cms.bool(True),
    EnableDTMeasurement = cms.bool(True),
    ForcePointDown = cms.bool(False),
    MaxCSCChi2 = cms.double(300.0),
    MaxDTChi2 = cms.double(300.0),
    MaxSeeds = cms.int32(1000)
)
