import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C6_cff import Phase2C6
process = cms.Process('DIGI',Phase2C6)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D44Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D44_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
     wantSummary = cms.untracked.bool(False),
     numberOfThreads = cms.untracked.uint32(8),
     numberOfStreams = cms.untracked.uint32(8)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

# this is for the test of the V9 Geo
process.FEVTDEBUGoutputGEO = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
                                           fileName = cms.untracked.string('file:/tmp/dalfonso/junk.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

##################

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring(
        'keep *_genParticles_*_*',
        'keep *_hgcalBackEndLayer1Producer_*_*',
        'keep *_hgcalBackEndLayer2Producer_*_*',
        'keep *_hgcalTowerProducer_*_*',
    ),
    fileName = cms.untracked.string('file:/tmp/dalfonso/test.root')
)

##################

# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("/tmp/dalfonso/test_triggergeom.root")
    )


# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
        MinEta = cms.double(3.499),
        MaxEta = cms.double(3.501),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(19.999),
        MaxPt = cms.double(20.001),
# 13 muon and 11 ele works, pions 211 no
#        ParticleID = cms.vint32(211)
        ParticleID = cms.vint32(22)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single pion pt 10 eta 35')
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutputGEO)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
# Eventually modify default geometry parameters
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V9
process = custom_geometry_V9(process, 2)

process.hgcaltriggergeomtester = cms.EDAnalyzer(
    "HGCalTriggerGeomTesterV9Imp2"
    )
process.test_step = cms.Path(process.hgcaltriggergeomtester)


## to test the full TP
process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)

# Schedule definition
# test_step
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.test_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# hgcl1tpg_step
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.hgcl1tpg_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
