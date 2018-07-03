import FWCore.ParameterSet.Config as cms

process = cms.Process("FSQDQM")
process.load("DQM.Physics.dqmfsq_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQM.collectorHost = ''
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.globaltag='92X_dataRun2_Prompt_v6'

#process.dqmSaver.workflow = cms.untracked.string('/Physics/FSQ/TESTFSQ')
process.dqmSaver.workflow = cms.untracked.string('workflow/for/mytest')
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/data/Run2017E/L1MinimumBias1/AOD/PromptReco-v1/000/303/728/00000/30D28B42-E5A1-E711-A6F1-02163E01A251.root')
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('detailedInfo'),
                                    detailedInfo = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
        threshold = cms.untracked.string('INFO')
        )
                                    )

#process.demo = cms.EDAnalyzer('FSQDQM')
process.p = cms.Path(process.FSQDQM)#+process.dqmSaver)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Validation.root")
                                   )
#process.p = cms.Path(process.demo)
