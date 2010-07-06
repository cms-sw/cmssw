import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# SUSYBSMAnalysis
# skim: SUSYBSM ElectronPhoton_HLT
process.load("SUSYBSMAnalysis.Skimming.SusyElectronPhoton_HLT_SkimPaths_cff")

process.load("SUSYBSMAnalysis.Skimming.susyHLTElectronPhotonOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDElectron_Skims7.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDElectron, file 3/3')
)
process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)
process.LoadAllDictionaries = cms.Service("LoadAllDictionaries")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/RelVal/2007/9/3/RelVal-RelValZTT-1188837971/0002/0279DD81-655A-DC11-8280-0016177CA778.root')
)

process.SUSYBSMAnalysis = cms.EndPath(process.susyHLTElectronPhotonOutputModule)

