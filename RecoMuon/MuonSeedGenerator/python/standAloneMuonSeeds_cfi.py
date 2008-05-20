import FWCore.ParameterSet.Config as cms

# Update this part asap!
# module standAloneMuonSeeds = MuonSeedGenerator {
MuonSeed = cms.EDFilter("MuonSeedGenerator",
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
    EnableDTMeasurement = cms.bool(True),
    EnableCSCMeasurement = cms.bool(True)
)



