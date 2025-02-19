import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *

from PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi import *
goodMuonMCMatch.src = 'userDataMuons'

from PhysicsTools.HepMCCandAlgos.goodTrackMCMatch_cfi import *
goodTrackMCMatch.src = 'userDataTracks'

from ElectroWeakAnalysis.Skimming.dimuonsMCMatch_cfi import *
#dimuonsMCMatch.src=cms.InputTag("userDataDimuons")

from ElectroWeakAnalysis.Skimming.dimuonsOneTrackMCMatch_cfi import *
#dimuonsOneTrackMCMatch.src=cms.InputTag("userDataDimuonsOneTrack")

#allDimuonsMCMatch = cms.EDFilter("GenParticleMatchMerger",
#    src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"), cms.InputTag("dimuonsMCMatch")),
#   filter = cms.bool(False) 
#)

allDimuonsMCMatch = cms.EDFilter("GenParticleMatchMerger",
   src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("dimuonsMCMatch")),
   filter = cms.bool(False)
)

allDimuonsOneTrackMCMatch = cms.EDFilter("GenParticleMatchMerger",
   src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"), cms.InputTag("dimuonsOneTrackMCMatch")),
   filter = cms.bool(False)
)

# Different MCtruth sequences for different ZMuMu paths
mcTruthForDimuons = cms.Sequence(goodMuonMCMatch+dimuonsMCMatch+allDimuonsMCMatch)
mcTruthForDimuonsOneTrack = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsOneTrackMCMatch+allDimuonsOneTrackMCMatch)

#mcTruthForDimuons = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsMCMatch+allDimuonsMCMatch)

#mcTruthForDimuonsOneTrack = cms.Sequence(goodMuonMCMatch+goodTrackMCMatch+dimuonsOneTrackMCMatch)


#dimuonsMCTruth = cms.Path(dimuonsHLTFilter+
#                          mcTruthForDimuons
#)

mcEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    ### MC matching infos
    'keep *_genParticles_*_*',
    'keep *_allDimuonsMCMatch_*_*',
    'keep *_allDimuonsOneTrackMCMatch_*_*'
     )
)
