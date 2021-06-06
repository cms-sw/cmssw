import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.ptSeedParameterization_cfi import ptSeedParameterization
from RecoMuon.MuonSeedGenerator.MuonSeedPtScale_cfi import dphiScale

import RecoMuon.MuonSeedGenerator.muonSeedGenerator_cfi as _mod
ancientMuonSeed = _mod.muonSeedGenerator.clone(
                                 ptSeedParameterization,
                                 dphiScale,
                                 beamSpotTag = "offlineBeamSpot",
                                 scaleDT = True,
                                 CSCRecSegmentLabel = "cscSegments",
                                 DTRecSegmentLabel  = "dt4DSegments",
                                 ME0RecSegmentLabel = "me0Segments",
                                 EnableDTMeasurement = True,
                                 EnableCSCMeasurement = True,
                                 EnableME0Measurement = False,
                                 # places where it's OK to have single-segment seeds
                                 crackEtas = [0.2, 1.6, 1.7],
                                 crackWindow = 0.04,
                                 deltaPhiSearchWindow = 0.25,
                                 deltaEtaSearchWindow = 0.2,
                                 deltaEtaCrackSearchWindow = 0.25,
                                 )

# phase2 ME0
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(ancientMuonSeed, EnableME0Measurement = True)
# phase2 GE0
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toModify(ancientMuonSeed, EnableME0Measurement = False)
