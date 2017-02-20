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
                                 GEMRecSegmentLabel = cms.InputTag("gemSegments"),
                                 GEMRecHitLabel = cms.InputTag("gemRecHits"),
                                 ME0RecSegmentLabel = cms.InputTag("me0Segments"),
                                 EnableDTMeasurement = cms.bool(True),
                                 EnableCSCMeasurement = cms.bool(True),
                                 EnableGEMMeasurement = cms.bool(False),
                                 EnableME0Measurement = cms.bool(False),
                                 # places where it's OK to have single-segment seeds
                                 crackEtas = cms.vdouble(0.2, 1.6, 1.7),
                                 crackWindow = cms.double(0.04),
                                 deltaPhiSearchWindow = cms.double(0.25),
                                 deltaEtaSearchWindow = cms.double(0.2),
                                 deltaEtaCrackSearchWindow = cms.double(0.25),
                                 )

# run3_GEM
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(ancientMuonSeed, EnableGEMMeasurement = cms.bool(True))

# phase2 ME0
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(ancientMuonSeed, EnableME0Measurement = cms.bool(True))
