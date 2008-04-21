import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from PhysicsTools.RecoAlgos.goodMuons_cfi import *
from PhysicsTools.RecoAlgos.goodTracks_cfi import *
from PhysicsTools.RecoAlgos.goodStandAloneMuonTracks_cfi import *
from PhysicsTools.IsolationAlgos.goodMuonIsolations_cfi import *
from PhysicsTools.IsolationAlgos.goodTrackIsolations_cfi import *
from PhysicsTools.IsolationAlgos.goodStandAloneMuonTrackIsolations_cfi import *
from PhysicsTools.IsolationAlgos.muonIsolations_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMu_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneStandAloneMuonTrack_cfi import *
from PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi import *
from PhysicsTools.HepMCCandAlgos.goodTrackMCMatch_cfi import *
from PhysicsTools.HepMCCandAlgos.goodStandAloneMuonTrackMCMatch_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuMCMatch_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneTrackMCMatch_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneStandAloneMuonTrackMCMatch_cfi import *
from ElectroWeakAnalysis.ZReco.goodZMCMatch_cfi import *
goodMuonRecoForZToMuMu = cms.Sequence(goodMuons+goodTracks*goodStandAloneMuonTracks+goodMuonIsolations+goodTrackIsolations+goodStandAloneMuonTrackIsolations+muonIsolations)
zToMuMuReco = cms.Sequence(goodZToMuMu+goodZToMuMuOneTrack+goodZToMuMuOneStandAloneMuonTrack)
mcTruthForZToMuMu = cms.Sequence(goodMuonMCMatch+goodZToMuMuMCMatch)
mcTruthForZToMuMuOneTrack = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+goodZToMuMuOneTrackMCMatch)
mcTruthForZToMuMuOneStandAloneMuonTrack = cms.Sequence(goodMuonMCMatch+goodStandAloneMuonTrackMCMatch+goodZToMuMuOneStandAloneMuonTrackMCMatch)

