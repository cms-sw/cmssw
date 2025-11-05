# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 -s DIGI:pdigi_valid --conditions auto:phase2_realistic_0T --datatier GEN-SIM-DIGI --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker,trackingPhase2PU140 -n 1 --filein file:step1.root --fileout file:step2.root --nThreads 4
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140

process = cms.Process('DIGI',phase2_tracker,trackingPhase2PU140)


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D500Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')

process.trackerGeometry.applyAlignment = False
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    #fileNames = cms.untracked.vstring('file:Geometry/TrackerCommonData/data/CRack_PhaseII/step1.root'),
    #replace this file with the file in your local directory.
    fileNames = cms.untracked.vstring('file:step1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

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
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)



# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')

# Drop any EDAlias that points to ECAL/HCAL digis from 'mix'
def _drop_aliases_pointing_to_mix_calo(process):
    keep = {"simSiPixelDigis", "simSiStripDigis"}
    for name, alias in list(process.aliases_().items()):
        print ("name is , ", name , "alias is ", alias )
        if name not in keep :
          delattr(process, name)
          print("[tracker-only] Removed EDAlias:", name)
    return process

process = _drop_aliases_pointing_to_mix_calo(process)


m = process.mix
cont = getattr(m, "digitizers", None) or getattr(m, "mixObjects", None)
for k in list(cont.parameters_().keys()):
    print ("k is " , k ) 
    if k not in ("tracker","pixel"):
        delattr(cont, k)

print("Top-level keys in process.mix:")
for name in process.mix.digitizers.pixel.SSDigitizerAlgorithm.parameters_().keys():
    print(" ", name)
print("----------------------------------")

algo = process.mix.digitizers.pixel.SSDigitizerAlgorithm

if hasattr(algo, "LorentzAngle_DB"):
        algo.LorentzAngle_DB = cms.bool(False)
if hasattr(algo, "TanLorentzAnglePerTesla_Barrel"):
        algo.TanLorentzAnglePerTesla_Barrel = cms.double(0.0)
if hasattr(algo, "TanLorentzAnglePerTesla_Endcap"):
        algo.TanLorentzAnglePerTesla_Endcap = cms.double(0.0)


process.digitisation_step = cms.Path( cms.SequencePlaceholder("randomEngineStateProducer")*process.mix)
process.clusterization_step = cms.Path(process.siPhase2Clusters)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
process.L1TrackTrigger.remove(process.ProducerDTC)
process.L1TrackTrigger.remove(process.L1TExtendedHybridTracks)
process.L1TrackTrigger.remove(process.L1THybridTracksWithAssociators)
process.L1TrackTrigger.remove(process.L1TPromptExtendedHybridTracksWithAssociators)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.clusterization_step , process.L1TrackTrigger_step,process.endjob_step,process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
