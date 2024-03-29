# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: EMTFTools/ParticleGuns/LLP_flatMass_cfi.py -n 1000 -s GEN,SIM,DIGI,L1TrackTrigger,L1 --nThreads 8 --conditions auto:phase2_realistic --era Phase2C17I13M9 --geometry Extended2026D88 --fileout file:out.root --eventcontent FEVTDEBUGHLT --pileup NoPileUp --beamspot HLLHC14TeV --datatier GEN-SIM-DIGI-RAW --customise SimGeneral/MixingModule/customiseStoredTPConfig.higherPtTP,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,L1Trigger/Configuration/customisePhase2TTOn110.customisePhase2TTOn110,L1Trigger/L1TMuonEndCapPhase2/config.customise_mc,EMTFTools/NtupleMaker/config.customise_ntuple --customise_commands=process.TFileService = cms.Service('TFileService', fileName = cms.string('out.root'))\nprocess.gemRecHits.gemDigiLabel = 'simMuonGEMDigis'\nprocess.emtfToolsNtupleMaker.EMTFP2SimInfoEnabled = cms.bool(False)\nprocess.emtfToolsNtupleMaker.L1TrackTriggerTracksEnabled = cms.bool(False)\nprocess.emtfToolsNtupleMaker.TrackingParticlesEnabled = cms.bool(True)\nprocess.emtfToolsNtupleMaker.GenParticlesEnabled = cms.bool(False) --python_filename test/phase2/pset_XTo2LongLivedTo4Mu.py --no_exec --mc
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('L1',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100),
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
    annotation = cms.untracked.string('EMTFTools/ParticleGuns/LLP_flatMass_cfi.py nevts:1000'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('FEVTDEBUGHLT'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('emtfwithgmt.root'),
    outputCommands = cms.untracked.vstring(
        "drop *_*_*_*",
        'keep *_l1tStubsGmt_*_*',
        "keep *_l1tFwdMuonsGmt_*_*",
        "keep *_genParticles_*_*",
        "keep *_l1tTTTracksFromTrackletEmulation_Level1TTTracks_*",
        "keep *_l1tTkMuons_*_*"
    ),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.generator = cms.EDProducer("FlatRandomLLPGunProducer2",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        LLPMassSpectrum = cms.string('flatMass'),
        MaxCTauLLP = cms.double(5000),
        MaxEta = cms.double(3.5),
        MaxMassH = cms.double(1000),
        MaxPhi = cms.double(3.141592653589793),
        MaxPtH = cms.double(120),
        MinCTauLLP = cms.double(10),
        MinEta = cms.double(1e-06),
        MinMassH = cms.double(20),
        MinPhi = cms.double(-3.141592653589793),
        MinPtH = cms.double(1),
        PartID = cms.vint32(-13)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('LLP decay with flat mass spectrum')
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# EMTF++ Step
process.load('L1Trigger.L1TMuonEndCapPhase2.simCscTriggerPrimitiveDigisForEMTF_cfi')
process.load('L1Trigger.L1TMuonEndCapPhase2.rpcRecHitsForEMTF_cfi')
process.load('L1Trigger.L1TMuonEndCapPhase2.simEmtfDigisPhase2_cfi')

process.gemRecHits.gemDigiLabel = 'simMuonGEMDigis'
process.simEmtfDigisPhase2.Verbosity = cms.untracked.int32(1)

process.L1TMuonEndCapPhase2Task = cms.Task(
            process.simCscTriggerPrimitiveDigisForEMTF,
            process.rpcRecHitsForEMTF,
            process.simEmtfDigisPhase2
)
process.L1TMuonEndCapPhase2Sequence = cms.Sequence(process.L1TMuonEndCapPhase2Task)
process.L1TMuonEndCapPhase2_step = cms.Path(process.L1TMuonEndCapPhase2Sequence)


# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1TrackTrigger_step,process.L1simulation_step,process.L1TMuonEndCapPhase2_step, process.endjob_step,process.FEVTDEBUGHLToutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.customiseStoredTPConfig
from SimGeneral.MixingModule.customiseStoredTPConfig import higherPtTP

#call to customisation function higherPtTP imported from SimGeneral.MixingModule.customiseStoredTPConfig
process = higherPtTP(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.aging
from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000

#call to customisation function customise_aging_1000 imported from SLHCUpgradeSimulations.Configuration.aging
process = customise_aging_1000(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customisePhase2TTOn110
from L1Trigger.Configuration.customisePhase2TTOn110 import customisePhase2TTOn110

#call to customisation function customisePhase2TTOn110 imported from L1Trigger.Configuration.customisePhase2TTOn110
process = customisePhase2TTOn110(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
