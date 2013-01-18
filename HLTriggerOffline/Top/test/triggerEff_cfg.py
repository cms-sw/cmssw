import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTEffProcess")

process.load("HLTriggerOffline.Top.triggerEff_cfi")





process.source = cms.Source("PoolSource",
    #AlCaReco File
    fileNames = cms.untracked.vstring(
	'/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/194/702/EE71F090-EFA5-E111-84E0-BCAEC518FF52.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)





process.p = cms.Path(process.HLTEff)





