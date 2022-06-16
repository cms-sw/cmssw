import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.parseArguments()

## process def
process = cms.Process("TEST", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedRun3RoundOptics25ns13TeVLowSigmaZ_cfi')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("EventFilter.CSCRawToDigi.cscPacker_cfi")
process.load("EventFilter.CSCRawToDigi.cscPackerUnpackerUnitTestDef_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

## global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.XMLFromDBSource.label = cms.string("Extended")
process.genstepfilter.triggerConditions=cms.vstring("generation_step")

process.generator = cms.EDFilter(
        "Pythia8PtGun",
        PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
                MaxEta = cms.double(2.4),
                MaxPhi = cms.double(3.14159265359),
                MaxPt = cms.double(1000.1),
                MinEta = cms.double(0.9),
                MinPhi = cms.double(-3.14159265359),
                MinPt = cms.double(1.9),
                ParticleID = cms.vint32(-13)
        ),
        PythiaParameters = cms.PSet(
                parameterSets = cms.vstring()
        ),
        Verbosity = cms.untracked.int32(0),
        firstRun = cms.untracked.uint32(1),
        psethack = cms.string('single mu pt 1')
)


process.FEVTDEBUGHLToutput = cms.OutputModule(
        "PoolOutputModule",
        SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('generation_step')
        ),
        dataset = cms.untracked.PSet(
                dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
                filterName = cms.untracked.string('')
        ),
        fileName = cms.untracked.string('file:step1.root'),
        outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
        splitLevel = cms.untracked.int32(0)
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)

process.muonCSCDigis.InputObjects = "cscpacker:CSCRawData"

## specification of the test with the packer expert settings
process.cscPackerUnpackerUnitTestDef.usePreTriggers = process.cscpacker.usePreTriggers
process.cscPackerUnpackerUnitTestDef.packEverything = process.cscpacker.packEverything
process.cscPackerUnpackerUnitTestDef.packByCFEB = process.cscpacker.packByCFEB
process.cscPackerUnpackerUnitTestDef.formatVersion = process.cscpacker.formatVersion
process.cscPackerUnpackerUnitTestDef.useCSCShowers = process.cscpacker.useCSCShowers

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.simMuonGEMPadDigis * process.simMuonGEMPadDigiClusters * process.simCscTriggerPrimitiveDigis)
process.testPackUnpack_step = cms.Path(process.cscpacker * process.muonCSCDigis * process.cscPackerUnpackerUnitTestDef)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step, process.testPackUnpack_step, process.endjob_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)
