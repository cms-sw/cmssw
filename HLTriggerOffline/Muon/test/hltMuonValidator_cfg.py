import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("HLTriggerOffline.Muon.hltMuonValidator_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

##############################################################
##### Templates to change parameters in hltMuonValidator #####
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.processNameHlt  = "HLT"
# process.hltMuonValidator.cutMotherId     = 24
# process.hltMuonValidator.cutMinPt        = 10.0
# process.hltMuonValidator.cutMaxEta       =  2.1
##############################################################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/u0/zmm314debug.root'
    ),
    secondaryFileNames = cms.untracked.vstring(

    )
)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
process.MessageLogger.destinations += ['HLTMuonValMessages']
process.MessageLogger.categories   += ['HLTMuonVal']
process.MessageLogger.debugModules += ['HLTMuonValidator']
process.MessageLogger.HLTMuonValMessages = cms.untracked.PSet(
    threshold  = cms.untracked.string('DEBUG'),
    default    = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    HLTMuonVal = cms.untracked.PSet(limit = cms.untracked.int32(1000))
    )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTMuonOfflineAnalysis'),
    fileName = cms.untracked.string('hltMuonValidator.root')
)

process.analyzerpath = cms.Path(
    process.hltMuonValidator *
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.out)
