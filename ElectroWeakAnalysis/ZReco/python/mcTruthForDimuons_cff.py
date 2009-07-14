import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi import *
goodMuonMCMatch.src = 'selectedLayer1Muons'
#goodMuonMCMatch.src = 'selectedLayer1MuonsTriggerMatch'
from PhysicsTools.HepMCCandAlgos.goodTrackMCMatch_cfi import *
goodTrackMCMatch.src = 'selectedLayer1TrackCands'

from ElectroWeakAnalysis.ZReco.dimuonsMCMatch_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrackMCMatch_cfi import *
allDimuonsMCMatch = cms.EDFilter("GenParticleMatchMerger",
    src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"), cms.InputTag("dimuonsMCMatch"), cms.InputTag("dimuonsOneTrackMCMatch"))
)

mcTruthForDimuons = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsMCMatch+dimuonsOneTrackMCMatch+allDimuonsMCMatch)

