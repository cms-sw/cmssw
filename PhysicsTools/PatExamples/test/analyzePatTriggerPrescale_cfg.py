import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/user/vadler/cms/PatTutorial/CMSSW_5_2_5/data/patTrigger_dataFromRAW.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'analyzePatTriggerPrescale.root' )
)

process.triggerAnalysisPrescale = cms.EDAnalyzer( "PatTriggerAnalyzerPrescale",
    pathName = cms.string( "HLT_HT450_v5" )
)

process.p = cms.Path(
    process.triggerAnalysisPrescale
)
