import FWCore.ParameterSet.Config as cms

process = cms.Process('TestVertex')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200000),
    # input = cms.untracked.int32(100),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("EmptySource")

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:step1.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(2.85),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(100.0),
        MinEta = cms.double(-2.85),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(2.0),
        ParticleID = cms.vint32(-13)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single mu pt 2 to 100')
)

process.vtxtester = cms.EDAnalyzer("VtxTester")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('vtxTester.root')
)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen+process.vtxtester)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)

process.MessageLogger.cerr.FwkReport.reportEvery = 10000
