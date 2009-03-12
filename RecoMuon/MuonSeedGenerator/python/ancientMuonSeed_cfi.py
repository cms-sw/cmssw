import FWCore.ParameterSet.Config as cms

# Update this part asap!
# Old field map
#include "RecoMuon/MuonSeedGenerator/data/ptSeedParameterization_40T_851.cfi"
# New map at 3.8 T
#from RecoMuon.MuonSeedGenerator.ptSeedParameterization_cfi import *
from RecoMuon.MuonSeedGenerator.ptSeedParameterization_31X_cfi import *
from RecoMuon.MuonSeedGenerator.MuonSeedPtScale_cfi import *

# module standAloneMuonSeeds = MuonSeedGenerator {
ancientMuonSeed = cms.EDProducer("MuonSeedGenerator",
                                 ptSeedParameterization,
                                 dphiScale,
                                 scaleDT = cms.bool(True),
                                 CSCRecSegmentLabel = cms.InputTag("cscSegments"),
                                 DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
                                 EnableDTMeasurement = cms.bool(True),
                                 EnableCSCMeasurement = cms.bool(True),
                                 # places where it's OK to have single-segment seeds
                                 crackEtas = cms.vdouble(0.2, 1.6, 1.7),
                                 crackWindow = cms.double(0.04)
                                 )



