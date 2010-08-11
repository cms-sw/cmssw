import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DiffractiveForwardAnalysis
# skim: gammagammaEE
process.load("DiffractiveForwardAnalysis.Skimming.gammagammaEE_SkimPaths_cff")

process.load("DiffractiveForwardAnalysis.Skimming.gammagammaEEOutputModule_cfi")

# ElectroWeakAnalysis
# skim: EWK Di-electron
process.load("ElectroWeakAnalysis.ZReco.zToEE_SkimPaths_cff")

# skim: EWK Single e
process.load("ElectroWeakAnalysis.WReco.wToENu_SkimPaths_cff")

process.load("ElectroWeakAnalysis.ZReco.zToEEOutputModule_cfi")

process.load("ElectroWeakAnalysis.WReco.wToENuOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.11 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDElectron_Skims1.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDElectron, file 2/2')
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

process.DiffractiveForwardAnalysis = cms.EndPath(process.gammagammaEEOutputModule)
process.ElectroWeakAnalysis = cms.EndPath(process.zToEEOutputModule+process.wToENuOutputModule)

