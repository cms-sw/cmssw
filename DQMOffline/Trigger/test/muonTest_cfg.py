import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


process = cms.Process("HLTMuonOfflineAnalysis")

#### Load packages
process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")

#### Process command-line arguments
options = VarParsing('analysis')
#options.setDefault('inputFiles', ' /store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v6-v1/10000/28B122AD-67E4-E511-9BDF-0CC47A4D766C.root')
options.setDefault('inputFiles', 
                   '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/44D4EAC0-4CE4-E511-BDC4-0CC47A4D75F0.root',
                   '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/9AA555F6-5CE4-E511-ACB9-0CC47A4D769E.root',
                    '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/A08C6EC8-4DE4-E511-B109-0025905A60DA.root',
                    '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/AA56D3B8-4CE4-E511-8B4E-002618FDA207.root',
                    '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/B8818F85-58E4-E511-BC34-0025905A610A.root',
                    '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/D69DC0C1-4CE4-E511-8C1A-0025905B85EE.root',
                    '/store/relval/CMSSW_8_0_1/RelValZMM_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v6-v1/10000/F0C5A7C0-4CE4-E511-80CC-0025905A6068.root'
                   )
options.setDefault('outputFile', './muonTest.root')

options.parseArguments()

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "auto:run2_hlt_GRun"



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(options.inputFiles),
)

process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    categories = cms.untracked.vstring('HLTMuonVal'),
    destinations = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_*'
    ),
    fileName = cms.untracked.string(options.outputFile),
)

process.analyzerpath = cms.Path(
    process.muonFullOfflineDQM *
#    process.dqmStoreStats *
    process.MEtoEDMConverter )

process.outpath = cms.EndPath(process.out)
