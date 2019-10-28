import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
process = cms.Process('recHitHF',Run2_25ns)
process.load('Calibration.HcalCalibAlgos.recAnalyzerHF_cfi')
process.load("FWCore.MessageService.MessageLogger_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('AnalyzerHF')

# import of standard configurations
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       'file:/uscms_data/d3/snandan/CMSSW_9_2_4/src/BAF988E1-2956-E711-B218-02163E019CD6.root',
        '/store/data/Run2017B/SingleElectron/RECO/PromptReco-v1/000/297/046/00000/BAF988E1-2956-E711-B218-02163E019CD6.root'
        )
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("SingleNeutrinoHF_Prereco.root")
                                   )


process.recAnalyzerHF.Ratio = cms.bool(False)
process.recAnalyzerHF.IgnoreL1 = cms.untracked.bool(True)
process.schedule = cms.Path(process.recAnalyzerHF)
