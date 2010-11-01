import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsSkimFilter = cms.EDFilter("HiggsToZZ4LeptonsSkimFilter",
 useHLT   = cms.untracked.bool(True), 
 HLTinst  = cms.string('higgsToZZ4LeptonsHLTAnalysis'),
 HLTflag  = cms.vstring('flaginput','flagHLTaccept'),
 Skiminst = cms.string('higgsToZZ4LeptonsSkimProducer'),
 Skimflag = cms.string('flagSkimaccept')
)


