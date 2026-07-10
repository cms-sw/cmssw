import FWCore.ParameterSet.Config as cms

# from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

process = cms.Process('SIM',Phase2C22I13M9)
# process = cms.Process('SIM',Phase2C22I13M9,dd4hep)

# process.Tracer = cms.Service('Tracer')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Geometry.MTDTBCommonData.GeometryExtendedMTDTB2025Reco_DUTinFront_cff')
# process.load('Geometry.MTDTBCommonData.GeometryDD4hepExtendedMTDTB2025Reco_DUTinFront_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load("IOMC.EventVertexGenerators.VtxSmearedBeamProfile_cfi")
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleMuFlatPt0p7To10_cfi nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step1.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')

process.VtxSmeared.MinEta = 0.8813735870195429
process.VtxSmeared.MaxEta = 0.8813735870195429
process.VtxSmeared.MinPhi = 0.
process.VtxSmeared.MaxPhi = 0.
process.VtxSmeared.BeamMeanX = 0.
process.VtxSmeared.BeamMeanY =  0.
process.VtxSmeared.BeamPosition = 0.
process.VtxSmeared.BeamSigmaX =  2.
process.VtxSmeared.BeamSigmaY =  2.

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(0.8813735870195429), # 45 degrees in eta
        MaxEta = cms.double(0.8813735870195429),
        MinPhi = cms.double(0.), # orthogonal to the chosen module
        MaxPhi = cms.double(0.),
        MinE   = cms.double(180.00),
        MaxE   = cms.double(180.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)


# Consider only MTD SDs

process.g4SimHits.OnlySDs = [
    'MtdSensitiveDetector'
]

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 1
process.options.numberOfStreams = 0

# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)

# Customisation from command line

process.g4SimHits.SteppingVerbosity = 1

process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            )
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            )
process.MessageLogger.cerr.VertexGenerator = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.G4cout = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.G4err = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.TimingSim = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.MtdSim = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.SimG4CoreApplication = cms.untracked.PSet(
            # limit = cms.untracked.int32(0)
            limit = cms.untracked.int32(-1)
            )
process.MessageLogger.cerr.TrackInformation = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            # limit = cms.untracked.int32(-1)
            )

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
