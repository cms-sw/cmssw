# The following comments couldn't be translated into the new config version:

#
# JetMET integration testing file for single fixed pt pions.
# Reconstruction for Jets and MET only.
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.JetMET.calorimetry_gen_Zprime_Dijets_700_cff")

# event vertex smearing - applies only once (internal check)
# Note : all internal generatoes will always do (0,0,0) vertex
#
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.JetMET.calorimetry_simulation_cff")

process.load("Configuration.JetMET.calorimetry_digitization_cff")

process.load("Configuration.JetMET.calorimetry_reconstruction_cff")

process.load("Configuration.JetMET.calorimetry_caltowers_cff")

process.load("Configuration.JetMET.calorimetry_jetmet_cff")

process.load("Configuration.JetMET.calorimetry_jetmet_gen_cff")

process.load("Configuration.JetMET.calorimetry_jetmetcorrections_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        hcalDigis = cms.untracked.uint32(1234),
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        ecalUnsuppressedDigis = cms.untracked.uint32(1234),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.TimerService = cms.Service("TimerService")

process.myTimer = cms.EDFilter("Timer",
    # whether to include timing info about Timer itself
    includeSelf = cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 'keep recoCaloJets_*_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoGenJets_*_*_*', 'keep recoGenMETs_*_*_*', 'keep *_genParticlesAllStableNoNu_*_*', 'keep *_genParticlesAllStable_*_*', 'keep *_genParticlesForMET_*_*', 'keep *_genParticleCandidates_*_*', 'keep *_towerMaker_*_*', 'keep *_caloTowers_*_*'),
    fileName = cms.untracked.string('calorimetry-gen-Zprime_Dijets_700.root')
)

process.p = cms.Path(process.VtxSmeared*process.simulation*process.caloDigi*process.caloReco*process.caloTowersRec*process.caloJetMet*process.caloJetMetGen*process.caloJetMetCorrections*process.myTimer)
process.outpath = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.p,process.outpath)

process.MessageLogger.cout.threshold = 'ERROR'
process.MessageLogger.cerr.default.limit = 10
process.VtxSmeared.SigmaX = 0.
process.VtxSmeared.SigmaY = 0.
process.VtxSmeared.SigmaZ = 0.

