import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi import *
goodMuonMCMatch.src = 'userDataMuons'

from PhysicsTools.HepMCCandAlgos.goodTrackMCMatch_cfi import *
goodTrackMCMatch.src = 'userDataTracks'

from ElectroWeakAnalysis.Skimming.dimuonsMCMatch_cfi import *
#dimuonsMCMatch.src=cms.InputTag("userDataDimuons")

from ElectroWeakAnalysis.Skimming.dimuonsOneTrackMCMatch_cfi import *
#dimuonsOneTrackMCMatch.src=cms.InputTag("userDataDimuonsOneTrack")

allDimuonsMCMatch = cms.EDFilter("GenParticleMatchMerger",
    src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"), cms.InputTag("dimuonsMCMatch")),
   filter = cms.bool(False) 
)




mcTruthForDimuons = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsMCMatch+allDimuonsMCMatch)

mcTruthForDimuonsOneTrack = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsOneTrackMCMatch)

