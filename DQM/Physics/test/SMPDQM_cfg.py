import FWCore.ParameterSet.Config as cms

process = cms.Process("SMPDQM")
process.load("DQM.Physics.dqmfsq_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQM.collectorHost = ''
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag.globaltag='92X_dataRun2_Prompt_v6'

#process.dqmSaver.workflow = cms.untracked.string('/Physics/SMP/TESTSMP')
process.dqmSaver.workflow = cms.untracked.string('workflow/for/mytest')
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_11_0_0_pre6/RelValSingleMuPt100_UP18/GEN-SIM-DIGI-RECO/110X_upgrade2018_realistic_v3_FastSim-v1/20000/EC423F2E-1E67-794B-8A88-AACBE3193AF3.root',
                                                              '/store/relval/CMSSW_11_0_0_pre6/RelValSingleMuPt100_UP18/GEN-SIM-DIGI-RECO/110X_upgrade2018_realistic_v3_FastSim-v1/20000/6C0A2520-1016-E74A-B1F3-CE94AF3C97ED.root'
                                                          )
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('detailedInfo'),
                                    detailedInfo = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
        threshold = cms.untracked.string('INFO')
        )
                                    )

#process.demo = cms.EDAnalyzer('SMPDQM')
process.p = cms.Path(process.SMPDQM)#+process.dqmSaver)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Validation.root")
                                   )
#process.p = cms.Path(process.demo)
