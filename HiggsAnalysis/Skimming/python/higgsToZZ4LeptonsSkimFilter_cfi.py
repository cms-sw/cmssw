import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsSkimFilter = cms.EDFilter("HiggsToZZ4LeptonsSkimFilter",

 # HLT
 useHLT   = cms.untracked.bool(False), 
 HLTinst  = cms.string('higgsToZZ4LeptonsHLTAnalysis'),
 HLTflag  = cms.vstring('flaginput','flagHLTaccept'),

 # DiLepton
 useDiLeptonSkim = cms.untracked.bool(True),
 SkimDiLeptoninst = cms.string('higgsToZZ4LeptonsSkimDiLeptonProducer'),
 SkimDiLeptonflag = cms.string('SkimDiLeptonA'),

 # TriLepton
 useTriLeptonSkim = cms.untracked.bool(True),
 SkimTriLeptoninst = cms.string('higgsToZZ4LeptonsSkimTriLeptonProducer'),
 SkimTriLeptonflag = cms.string('SkimTriLepton')   
)


