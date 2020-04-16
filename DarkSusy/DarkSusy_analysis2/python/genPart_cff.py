import FWCore.ParameterSet.Config as cms

process = cms.Process("prova")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/lustrehome/caime/data/DarkSusy-RunIIAutumn18DR_step3-MiniAOD.root'
    )
)

process.genan = cms.EDAnalyzer('DarkSusy_analysis2', 
                                    muonpat = cms.untracked.InputTag('slimmedMuons'), jetpat = cms.untracked.InputTag('slimmedJets'), trigger = cms.InputTag('TriggerResults','','HLT'))




process.TFileService = cms.Service("TFileService", fileName=cms.string("histo.root"))
process.p = cms.Path(process.genan)
