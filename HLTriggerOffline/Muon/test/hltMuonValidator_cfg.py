import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("HLTriggerOffline.Muon.HLTMuonVal_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

##############################################################################
##### Templates to change parameters in hltMuonValidator #####################
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.genMuonCut = "abs(mother.pdgId) == 24"
# process.hltMuonValidator.recMuonCut = "isGlobalMuon && eta < 1.2"
##############################################################################

hltProcessName = "HLT"
process.relvalMuonBits.TriggerResultsTag.setProcessName(hltProcessName)
process.hltMuonValidator.hltProcessName = hltProcessName

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
        'file:/u0/zmm314debug.root'
    ),
    secondaryFileNames = cms.untracked.vstring(

    )
)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.MessageLogger.debugModules += ['HLTMuonValidator']
process.MessageLogger.files.HLTMuonValMessages = cms.untracked.PSet(
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
    process.relvalMuonBits *
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.out)
