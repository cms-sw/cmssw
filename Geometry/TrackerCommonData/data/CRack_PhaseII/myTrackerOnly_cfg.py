# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: UndergroundCosmicSPLooseMu_cfi -s GEN,SIM -n 1 --conditions auto:phase2_realistic_0T --beamspot DBrealisticHLLHC --datatier GEN-SIM --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker,trackingPhase2PU140 --fileout file:step1.root --nThreads 4 --python_filename myTrackerOnly_cfg.py
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140

process = cms.Process('SIM',phase2_tracker,trackingPhase2PU140)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D500Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D500_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticHLLHC_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
'''
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),  # or 'INFO'
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),  # 0 = unlimited
        #INFO = cms.untracked.PSet(limit = cms.untracked.int32(100000)),
        #default = cms.untracked.PSet(limit = cms.untracked.int32(100000))
    ),
    debugModules = cms.untracked.vstring('*')  # Enable debug for all modules
)
'''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000),
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
    annotation = cms.untracked.string('UndergroundCosmicSPLooseMu_cfi nevts:1'),
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

process.generator = cms.EDProducer("CosMuoGenProducer",
    MaxP = cms.double(3000.0),            # Max momentum [GeV]
    MinP = cms.double(5.0),               # Min momentum [GeV]
    MinEnu = cms.double(5.0),
    MaxEnu = cms.double(10000.0),

    MinTheta = cms.double(0.0),          # Steep angle (close to horizontal)
    MaxTheta = cms.double(85.0),         # Allow small variation

    MinPhi = cms.double(0.0),
    MaxPhi = cms.double(360.0),           # Full azimuthal coverage

    ZCentrOfTarget = cms.double(00.0),  # CRACK z-center [mm]
    ZDistOfTarget  = cms.double(0.0),   # Extend z-window [mm]
    RadiusOfTarget = cms.double(8000.0),   # Enough to cover module spread

    PlugVx = cms.double(0.0),             # Target centered at (0,0)
    PlugVz = cms.double(-14000.0),        # Entry point upstream (long propagation)

    AcptAllMu = cms.bool(False),           # Accept all muons (no barrel bias)
    #MultiMuon = cms.bool(False),          # Disable multi-muon mode
    TIFOnly_constant = cms.bool(False),   # You can set to True if needed
    TIFOnly_linear = cms.bool(False),  # <-- MISSING before
    TrackerOnly = cms.bool(True),        # Avoid tracker-only filtering

    Verbosity = cms.bool(False),           # Enable for debugging

    # Optional tweaks
    MaxT0 = cms.double(25.0),
    MinT0 = cms.double(-25.0),
    RhoAir = cms.double(0.001214),
    MinP_CMS = cms.double(-1.0),
    ElossScaleFactor = cms.double(1.0),
    MultiMuon = cms.bool(False),
    MultiMuonFileFirstEvent = cms.int32(1),
    MultiMuonFileName = cms.string("CORSIKAmultiMuon.root"),
    MultiMuonNmin = cms.int32(2),
    MTCCHalf = cms.bool(False),
    RhoClay = cms.double(0.0),
    RhoPlug = cms.double(0.0),
    RhoRock = cms.double(0.0),
    RhoWall = cms.double(0.0),
    ClayWidth = cms.double(0.0),
    NuProdAlt = cms.double(5000.0),    # production altitude in mm

)

'''
process.cosmicInPixelLoose = cms.EDFilter("CosmicGenFilterHelix",
    charges = cms.vint32(1, -1),
    doMonitor = cms.untracked.bool(False),
    maxZ = cms.double(100.0),
    minP = cms.double(0.0),
    minPt = cms.double(0.0),
    minZ = cms.double(-100.0),
    pdgIds = cms.vint32(-13, 13),
    propagator = cms.string('SteppingHelixPropagatorAlong'),
    radius = cms.double(20.0),
    src = cms.InputTag("generator","unsmeared")
)


process.cosmicInTracker = cms.EDFilter("CosmicGenFilterHelix",
    charges = cms.vint32(1, -1),
    doMonitor = cms.untracked.bool(False),
    maxZ = cms.double(212.0),
    minP = cms.double(0.0),
    minPt = cms.double(0.0),
    minZ = cms.double(-212.0),
    pdgIds = cms.vint32(-13, 13),
    propagator = cms.string('SteppingHelixPropagatorAlong'),
    radius = cms.double(100.0),
    src = cms.InputTag("generator","unsmeared")
)


process.SteppingHelixPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
    ApplyRadX0Correction = cms.bool(True),
    AssumeNoMaterial = cms.bool(False),
    ComponentName = cms.string('SteppingHelixPropagatorAlong'),
    NoErrorPropagation = cms.bool(False),
    PropagationDirection = cms.string('alongMomentum'),
    SetVBFPointer = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    debug = cms.bool(False),
    endcapShiftInZNeg = cms.double(0.0),
    endcapShiftInZPos = cms.double(0.0),
    returnTangentPlane = cms.bool(True),
    sendLogWarning = cms.bool(False),
    useEndcapShiftsInZ = cms.bool(False),
    useInTeslaFromMagField = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    useMagVolumes = cms.bool(True),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False)
)
'''

#process.ProductionFilterSequence = cms.Sequence(process.generator+process.cosmicInPixelLoose)
process.ProductionFilterSequence = cms.Sequence(process.generator)

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
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.eventSetup.numberOfConcurrentIOVs = 1
process.g4SimHits.Verbosity = cms.untracked.int32(5)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.ProductionFilterSequence)

process.g4SimHits.TrackingCut = cms.PSet(
    EkinMin = cms.double(1.0)  # in MeV
)
from SimG4Core.Configuration.customise_trackerOnlySim import customise
process = customise(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

