import FWCore.ParameterSet.Config as cms

process = cms.Process('EvtAna')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
			    fileNames = cms.untracked.vstring('file:test.root'),
)

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '75X_dataRun2_v6', '')

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi") 

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string("eventtree_filtered_data.root"))


process.load('HeavyIonsAnalysis.Configuration.collisionEventSelection_cff')

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')

process.p = cms.Path(process.collisionEventSelectionAOD * process.centralityBin * process.hiEvtAnalyzer)
