import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTSMPOfflineAnalysis")

process.load("HLTriggerOffline.SMP.SMPValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

hltProcessName = "HLT"
process.hltSMPValidator.hltProcessName = hltProcessName

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string(autoCond['startup'])

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_3_0_pre1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/A8F284E4-FC59-E411-8934-0025905A48D0.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/F2BA47E7-FC59-E411-9031-0025905964B4.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
    )
)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
process.MessageLogger.destinations += ['SMPValidationMessages']
process.MessageLogger.categories   += ['SMPValidation']
process.MessageLogger.debugModules += ['*']#HLTHiggsValidator','HLTHiggsSubAnalysis','HLTHiggsPlotter']
process.MessageLogger.SMPValidationMessages = cms.untracked.PSet(
    threshold       = cms.untracked.string('DEBUG'),
    default         = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    SMPValidation = cms.untracked.PSet(limit = cms.untracked.int32(1000))
    )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTSMPOfflineAnalysis'),
    fileName = cms.untracked.string('hltSMPValidator.root')
)


process.analyzerpath = cms.Path(
    process.hltSMPValidator *
    process.MEtoEDMConverter
)


process.outpath = cms.EndPath(process.out)
