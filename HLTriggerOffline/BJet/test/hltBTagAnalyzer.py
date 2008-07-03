import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTValidation")
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("HLTriggerOffline.BJet.hltJetMCTools_cff")

process.load("HLTriggerOffline.BJet.hltBLifetimeExtra_cff")

process.load("HLTriggerOffline.BJet.hltBLifetime_cff")

process.load("HLTriggerOffline.BJet.hltBLifetimeRelaxed_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuonExtra_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuon_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuonRelaxed_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.extra_lifetime = cms.Path(process.hltBLifetimeExtra)
process.extra_lifetime_rx = cms.Path(process.hltBLifetimeExtraRelaxed)
process.extra_softmuon = cms.Path(process.hltBSoftmuonExtra)
process.extra_softmuon_rx = cms.Path(process.hltBSoftmuonExtraRelaxed)
process.extra_softmuon_dr = cms.Path(process.hltBSoftmuonExtraByDR)
process.extra_jetmctools = cms.Path(process.hltJetMCTools)
process.validation = cms.Path(process.hltBLifetime+process.hltBLifetimeRelaxed+process.hltBSoftmuon+process.hltBSoftmuonRelaxed)
process.schedule = cms.Schedule(process.extra_lifetime,process.extra_lifetime_rx,process.extra_softmuon,process.extra_softmuon_rx,process.extra_softmuon_dr,process.extra_jetmctools,process.validation)

process.MessageLogger.categories.append('HLTBtagAnalyzer')
process.MessageLogger.debugModules.extend(process.hltBLifetime_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBLifetimeRelaxed_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuon_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuonRelaxed_modules.modules)

