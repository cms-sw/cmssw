# The following comments couldn't be translated into the new config version:

# modified version of hltBtagLifetimeAnalyzer.cfg, running on SIM+HLT only
# - MC-matching info are built on the fly
# - Offline-matching is disabled

import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTValidation")
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.HepMCCandAlgos.genParticleCandidates_cfi")

process.load("HLTriggerOffline.BJet.hltJetMCTools_cff")

process.load("HLTriggerOffline.BJet.hltBLifetimeExtra_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuonExtra_cff")

process.load("HLTriggerOffline.BJet.hltBLifetime_cff")

process.load("HLTriggerOffline.BJet.hltBSoftmuon_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.extra_lifetime = cms.Path(process.hltBLifetimeExtra)
process.extra_softmuon = cms.Path(process.hltBSoftmuonExtra)
process.extra_softmuon_dr = cms.Path(process.hltBSoftmuonExtraByDR)
process.extra_jetmctools = cms.Path(process.genParticleCandidates*process.hltJetMCTools)
process.validation = cms.Path(process.hltBLifetime+process.hltBSoftmuon)
process.schedule = cms.Schedule(process.extra_lifetime,process.extra_softmuon,process.extra_softmuon_dr,process.extra_jetmctools,process.validation)

process.MessageLogger.categories.append('HLTBtagAnalyzer')
process.MessageLogger.debugModules.extend(process.hltBLifetime_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuon_modules.modules)
process.hltb1jet.offlineBJets = 'none'
process.hltb2jet.offlineBJets = 'none'
process.hltb3jet.offlineBJets = 'none'
process.hltb4jet.offlineBJets = 'none'
process.hltbht.offlineBJets = 'none'
process.hltb1jetmu.offlineBJets = 'none'
process.hltb2jetmu.offlineBJets = 'none'
process.hltb3jetmu.offlineBJets = 'none'
process.hltb4jetmu.offlineBJets = 'none'
process.hltbhtmu.offlineBJets = 'none'

