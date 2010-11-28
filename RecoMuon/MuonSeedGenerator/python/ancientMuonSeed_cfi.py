import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.ptSeedParameterization_cfi import *
from RecoMuon.MuonSeedGenerator.MuonSeedPtScale_cfi import *

# module standAloneMuonSeeds = MuonSeedGenerator {
ancientMuonSeed = cms.EDProducer("MuonSeedGenerator",
                                 ptSeedParameterization,
                                 dphiScale,
                                 beamSpotTag = cms.InputTag("offlineBeamSpot"),
                                 scaleDT = cms.bool(True),
                                 CSCRecSegmentLabel = cms.InputTag("cscSegments"),
                                 DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
                                 EnableDTMeasurement = cms.bool(True),
                                 EnableCSCMeasurement = cms.bool(True),
                                 # places where it's OK to have single-segment seeds
                                 crackEtas = cms.vdouble(0.2, 1.6, 1.7),
                                 crackWindow = cms.double(0.04)
                                 )



