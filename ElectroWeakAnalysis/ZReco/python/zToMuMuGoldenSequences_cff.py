import FWCore.ParameterSet.Config as cms

from PhysicsTools.RecoAlgos.allTracks_cfi import *
from PhysicsTools.RecoAlgos.allMuons_cfi import *
from PhysicsTools.IsolationAlgos.allMuonIsolations_cfi import *
from PhysicsTools.HepMCCandAlgos.allMuonsGenParticlesMatch_cfi import *
from PhysicsTools.HepMCCandAlgos.allTracksGenParticlesLeptonMatch_cfi import *
import ElectroWeakAnalysis.ZReco.zToMuMu_cfi
zToMuMuGolden = ElectroWeakAnalysis.ZReco.zToMuMu_cfi.zToMuMu.clone()
import ElectroWeakAnalysis.ZReco.zToMuMuGenParticlesMatch_cfi
zToMuMuGoldenGenParticlesMatch = ElectroWeakAnalysis.ZReco.zToMuMuGenParticlesMatch_cfi.zToMuMuGenParticlesMatch.clone()
muonRecoForZToMuMuGolden = cms.Sequence(allTracks+allMuons+allMuonIsolations)
zToMuMuGoldenAnalysisSequenceData = cms.Sequence(muonRecoForZToMuMuGolden+zToMuMuGolden)
muonMCTruthForZToMuMuGolden = cms.Sequence(allMuonsGenParticlesMatch+allTracksGenParticlesLeptonMatch+zToMuMuGoldenGenParticlesMatch)
zToMuMuGoldenAnalysisSequence = cms.Sequence(zToMuMuGoldenAnalysisSequenceData+muonMCTruthForZToMuMuGolden)
zToMuMuGolden.massMin = 40
zToMuMuGoldenGenParticlesMatch.src = 'zToMuMuGolden'

