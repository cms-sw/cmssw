import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTValidation")
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("HLTriggerOffline.BJet.hltBLifetimeAOD_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuonAOD_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.validation = cms.Path(process.hltBLifetime+process.hltBSoftmuon)
process.MessageLogger.categories.append('HLTBtagAnalyzer')
process.MessageLogger.debugModules.extend(process.hltBLifetime_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuon_modules.modules)

