# Runs the pat::GenJet chain and the three NanoAOD tables on the generated sample
# (PhysicsTools/PatAlgos/test/patGenJet_gen_ttbar20_cfg.py), together with today's gen tables (GenJet, GenJetAK8,
# SubGenJetAK8) for the comparison of patGenJet_compare_gen_tables.py.
#
#   cmsRun run_patgenjets_cfg.py input=ttbar20_gen.root output=ttbar20_nano.root [edm=ttbar20_pat.root]
#
# Offline: no GlobalTag, no conditions. The standard NANO step cannot run on the GEN-only sample, so the process
# is built in Python: era, nano_cff and nanoAOD_customizeCommon only define the table modules whose columns are copied;
# only the modules of the path below run.
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("input", "ttbar20_gen.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "generated sample")
options.register("output", "ttbar20_nano.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "NanoAOD file")
options.register("edm", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "optional EDM file with the pat::GenJet products")
options.parseArguments()

from Configuration.Eras.Era_Run3_2026_cff import Run3_2026
process = cms.Process("PATGENNANO", Run3_2026)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:" + options.input))

# NANO table definitions (columns of the running process)
process.load("PhysicsTools.NanoAOD.nano_cff")
from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon
process = nanoAOD_customizeCommon(process)

# what MiniAOD makes and the pat::GenJet chain reads: pruned/packed gen particles (re-made from genParticles, as in MiniAOD) and today's
# slimmed gen jets (for the comparison and for the forward links of genJetFwdRef); ak8GenJetsNoNuSoftDrop comes from the input file
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")  # the pruner reads the particle data table
process.load("PhysicsTools.PatAlgos.slimming.genParticles_cff")
from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJetsSoftDrop
process.ak8GenJetsNoNuSoftDrop = ak8GenJetsSoftDrop.clone(src=cms.InputTag("ak8GenJetsNoNuConstituents", "constituents"))  # not scheduled
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi import slimmedGenJets, slimmedGenJetsAK8
process.slimmedGenJets = slimmedGenJets.clone()
process.slimmedGenJetsAK8 = slimmedGenJetsAK8.clone()
process.slimmedGenJetsAK8SoftDropSubJets = cms.EDProducer("PATGenJetSlimmer",
    src=cms.InputTag("ak8GenJetsNoNuSoftDrop", "SubJets"), packedGenParticles=cms.InputTag("packedGenParticles"),
    cut=cms.string(""), cutLoose=cms.string(""), nLoose=cms.uint32(0), clearDaughters=cms.bool(False), dropSpecific=cms.bool(True))

# pat::GenJet: chain (as MiniAOD with customisePatGenJetsMiniAOD) and tables
from PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD import customisePatGenJetsMiniAOD
process = customisePatGenJetsMiniAOD(process)
from PhysicsTools.NanoAOD.customisePatGenJetsNano import customisePatGenJetsNano
process = customisePatGenJetsNano(process)

# Test table: the AK8 table without cut, so that fat jets without a soft-drop match (no groomed ECF user floats) are evaluated
process.patGenJetAK8AllTable = process.patGenJetAK8Table.clone(cut="", name="PatGenJetAK8All")

process.patGenTask = cms.Task(
    process.prunedGenParticlesWithStatusOne, process.prunedGenParticles, process.packedGenParticles,
    process.patGenJetsTask,
    process.slimmedGenJets, process.slimmedGenJetsAK8, process.slimmedGenJetsAK8SoftDropSubJets,
    process.genJetTable, process.genJetAK8Table, process.genSubJetAK8Table,
    process.patGenJetTablesTask, process.patGenJetAK8AllTable)
process.p = cms.Path(process.patGenTask)
process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule", fileName=cms.untracked.string(options.output),
    outputCommands=cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*Table_*_*"),
    compressionAlgorithm=cms.untracked.string("LZMA"), compressionLevel=cms.untracked.int32(9))
process.e = cms.EndPath(process.NANOAODSIMoutput)
if options.edm:
    process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string(options.edm),
        outputCommands=cms.untracked.vstring("drop *", "keep *_patGenJets*_*_*", "keep *_selectedPatGenJetsAK8_*_*", "keep *_packedPatGenJetsAK8_*_*"))
    process.e2 = cms.EndPath(process.out)
