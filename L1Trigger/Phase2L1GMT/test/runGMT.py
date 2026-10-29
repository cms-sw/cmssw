# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step1 --conditions 111X_mcRun4_realistic_T15_v3 -n 2 --era Phase2C9 --eventcontent FEVTDEBUGHLT --runUnscheduled file:/eos/cms/store/relval/CMSSW_11_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v2/10000/01054EE2-1B51-C449-91A2-5202A60D16A3.root -s RAW2DIGI,L1TrackTrigger,L1 --datatier FEVTDEBUGHLT --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,L1Trigger/Configuration/customisePhase2TTNoMC.customisePhase2TTNoMC,Configuration/DataProcessing/Utils.addMonitoring --geometry Extended2026D49 --fileout file:/tmp/step1_Reprocess_TrackTrigger_L1.root --no_exec --nThreads 8 --python step1_L1_ProdLike.py --filein das:/TT_TuneCP5_14TeV-powheg-pythia8/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("L1", Phase2C17I13M9)

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D121Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.L1TrackTrigger_cff")
process.load("Configuration.StandardSequences.SimL1Emulator_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(-1),
    output=cms.optional.untracked.allowed(cms.int32, cms.PSet),
)

# Input source
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        "/store/relval/CMSSW_20_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D128_RegeneratedGS_PU_20260903_192421_RV19-v1/2830000/653dcc08-10c5-497e-8394-f31096868d42.root"
    ),
    inputCommands=cms.untracked.vstring(
        "keep *",
        "drop l1tPFJets_*_*_*",
        "drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT",
        "drop triggerTriggerFilterObject*_*_*_HLT",
    ),
    secondaryFileNames=cms.untracked.vstring(),
)


process.options = cms.untracked.PSet(
    IgnoreCompletely=cms.untracked.vstring(),
    Rethrow=cms.untracked.vstring(),
    TryToContinue=cms.untracked.vstring(),
    accelerators=cms.untracked.vstring("*"),
    allowUnscheduled=cms.obsolete.untracked.bool,
    canDeleteEarly=cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules=cms.untracked.bool(True),
    dumpOptions=cms.untracked.bool(False),
    emptyRunLumiMode=cms.obsolete.untracked.string,
    eventSetup=cms.untracked.PSet(
        forceNumberOfConcurrentIOVs=cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs=cms.untracked.uint32(1),
    ),
    fileMode=cms.untracked.string("FULLMERGE"),
    forceEventSetupCacheClearOnNewRun=cms.obsolete.untracked.bool,
    holdsReferencesToDeleteEarly=cms.untracked.VPSet(),
    makeTriggerResults=cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue=cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly=cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(0),
    numberOfConcurrentRuns=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(0),
    numberOfThreads=cms.untracked.uint32(1),
    printDependencies=cms.untracked.bool(False),
    sizeOfStackForThreadsInKB=cms.optional.untracked.uint32,
    throwIfIllegalParameter=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation=cms.untracked.string("step1 nevts:2"),
    name=cms.untracked.string("Applications"),
    version=cms.untracked.string("$Revision: 1.19 $"),
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule(
    "PoolOutputModule",
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string("FEVTDEBUGHLT"),
        filterName=cms.untracked.string(""),
    ),
    fileName=cms.untracked.string("reprocess.root"),
    outputCommands=cms.untracked.vstring(
        "drop *_*_*_*",
        "keep *_*Gmt_*_*",
        "keep *_gmtStubs_*_*",
        "keep *_genParticles_*_*",
        "keep *_l1tTTTracksFromTrackletEmulation_Level1TTTracks_*",
        "keep *_l1tTkMuons_*_*",
    ),
    splitLevel=cms.untracked.int32(0),
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic", "")

# Path and EndPath definitions
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(
    process.L1simulation_step,
    process.endjob_step,
    process.FEVTDEBUGHLToutput_step,
)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask

associatePatAlgosToolsTask(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customisePhase2TTNoMC
from L1Trigger.Configuration.customisePhase2TTNoMC import customisePhase2TTNoMC
from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000

# call to customisation function customisePhase2TTNoMC imported from L1Trigger.Configuration.customisePhase2TTNoMC
process = customisePhase2TTNoMC(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring

# call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions

# Customisation from command line

process.source.inputCommands = cms.untracked.vstring(
    "keep *",
    "drop l1tPFJets_*_*_*",
    "drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT",
    "drop triggerTriggerFilterObject*_*_*_HLT",
    "drop *_hltIter2Phase2L3*_*_HLT",
)
# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete

process = customiseEarlyDelete(process)
# End adding early deletion
