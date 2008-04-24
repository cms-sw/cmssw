import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.JetMET.calorimetry_gen_particlegun_cff")

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
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('all-in-one-particlegun.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        simEcalUnsuppressedDigis = cms.untracked.uint32(1234),
        simHcalUnsuppressedDigis = cms.untracked.uint32(1234),
        simHcalDigis = cms.untracked.uint32(1234),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p1 = cms.Path(process.VtxSmeared*process.simulation*process.caloDigi*process.caloReco*process.caloTowersRec*process.caloJetMet*process.caloJetMetCorrections+process.caloJetMetGen)
process.e = cms.EndPath(process.out)
process.MessageLogger.fwkJobReports = ['reco-application-calorimetry-all.log.xml']

