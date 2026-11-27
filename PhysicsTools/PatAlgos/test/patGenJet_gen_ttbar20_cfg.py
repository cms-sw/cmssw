# 20 Pythia8 ttbar events, produced offline (no input file, no GlobalTag, no conditions,
# no network, no vertex smearing module).
#
#   cmsRun patGenJet_gen_ttbar20_cfg.py output=ttbar20_gen.root
#
# Output (everything the pat::GenJet chain needs, nothing else):
#   genParticles, prunedGenParticlesWithStatusOne, prunedGenParticles (+ their Associations: the forward maps of
#   genCandidatesForwardMaps), packedGenParticles (+ its Association to genParticles),
#   genParticlesForJetsNoNu, ak4GenJetsNoNu, ak8GenJetsNoNu, ak8GenJetsNoNuConstituents, ak8GenJetsNoNuSoftDrop (+SubJets),
#   generator GenEventInfoProduct.
# The chain is the one MiniAOD runs: PhysicsTools/PatAlgos/python/slimming/genParticles_cff.py (pruned/packed) and
# applySubstructure_cff.py:34-35 (ak8GenJetsNoNuConstituents, ak8GenJetsNoNuSoftDrop).
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Generator.Pythia8CommonSettings_cfi import pythia8CommonSettingsBlock
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import pythia8CP5SettingsBlock

options = VarParsing("analysis")
options.register("output", "ttbar20_gen.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output EDM file")
options.register("events", 20, VarParsing.multiplicity.singleton, VarParsing.varType.int, "number of events")
options.register("seed", 12345, VarParsing.multiplicity.singleton, VarParsing.varType.int, "random seed")
options.parseArguments()

process = cms.Process("GENJETS")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.events))
process.source = cms.Source("EmptySource")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator=cms.PSet(initialSeed=cms.untracked.uint32(options.seed), engineName=cms.untracked.string("TRandom3")))
process.generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
    comEnergy=cms.double(13600.), maxEventsToPrint=cms.untracked.int32(0), pythiaPylistVerbosity=cms.untracked.int32(0),
    filterEfficiency=cms.untracked.double(1.0), pythiaHepMCVerbosity=cms.untracked.bool(False),
    PythiaParameters=cms.PSet(pythia8CommonSettingsBlock, pythia8CP5SettingsBlock,
        processParameters=cms.vstring("Top:gg2ttbar = on", "Top:qqbar2ttbar = on"),
        parameterSets=cms.vstring("pythia8CommonSettings", "pythia8CP5Settings", "processParameters")))

# genParticles from the unsmeared HepMC product (no vertex smearing module is loaded)
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.genParticles.src = cms.InputTag("generator", "unsmeared")

# generator jets as in AOD
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")
from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJetsSoftDrop, ak8GenJetsConstituents
process.ak8GenJetsNoNuConstituents = ak8GenJetsConstituents.clone(src="ak8GenJetsNoNu")
process.ak8GenJetsNoNuSoftDrop = ak8GenJetsSoftDrop.clone(src=cms.InputTag("ak8GenJetsNoNuConstituents", "constituents"))

# pruned and packed gen particles as in MiniAOD
process.load("PhysicsTools.PatAlgos.slimming.genParticles_cff")

process.p = cms.Path(process.generator * process.genParticles
                     * process.genParticlesForJetsNoNu * process.ak4GenJetsNoNu * process.ak8GenJetsNoNu
                     * process.ak8GenJetsNoNuConstituents * process.ak8GenJetsNoNuSoftDrop
                     * process.prunedGenParticlesWithStatusOne * process.prunedGenParticles * process.packedGenParticles)
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string(options.output),
    outputCommands=cms.untracked.vstring("drop *",
        "keep *_genParticles_*_*", "keep *_prunedGenParticlesWithStatusOne_*_*", "keep *_prunedGenParticles_*_*", "keep *_packedGenParticles_*_*",
        "keep *_genParticlesForJetsNoNu_*_*",
        "keep *_ak4GenJetsNoNu_*_*", "keep *_ak8GenJetsNoNu_*_*",
        "keep *_ak8GenJetsNoNuConstituents_*_*", "keep *_ak8GenJetsNoNuSoftDrop_*_*",
        "keep GenEventInfoProduct_generator_*_*"),
    SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring("p")))
process.e = cms.EndPath(process.out)
