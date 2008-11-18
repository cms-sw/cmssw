import FWCore.ParameterSet.Config as cms

# Update this part asap!
# Old field map
#include "RecoMuon/MuonSeedGenerator/data/ptSeedParameterization_40T_851.cfi"
# New map at 3.8 T
from RecoMuon.MuonSeedGenerator.ptSeedParameterization_38T_cfi import *
# New map at 4.0 T
#include "RecoMuon/MuonSeedGenerator/data/ptSeedParameterization_40T.cfi"
# module standAloneMuonSeeds = MuonSeedGenerator {
MuonSeed = cms.EDFilter("MuonSeedGenerator",
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
    EnableDTMeasurement = cms.bool(True),
    EnableCSCMeasurement = cms.bool(True)
)



