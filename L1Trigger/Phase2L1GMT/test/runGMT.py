# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --conditions 111X_mcRun4_realistic_T15_v3 -n 2 --era Phase2C9 --eventcontent FEVTDEBUGHLT --runUnscheduled file:/eos/cms/store/relval/CMSSW_11_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v2/10000/01054EE2-1B51-C449-91A2-5202A60D16A3.root -s RAW2DIGI,L1TrackTrigger,L1 --datatier FEVTDEBUGHLT --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,L1Trigger/Configuration/customisePhase2TTNoMC.customisePhase2TTNoMC,Configuration/DataProcessing/Utils.addMonitoring --geometry Extended2026D49 --fileout file:/tmp/step1_Reprocess_TrackTrigger_L1.root --no_exec --nThreads 8 --python step1_L1_ProdLike.py --filein das:/TT_TuneCP5_14TeV-powheg-pythia8/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('L1',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(18),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet),
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( (
#'/store/mc/Phase2HLTTDRWinter20DIGI/JPsiToMuMu_Pt0to100-pythia8_TuneCP5-gun/GEN-SIM-DIGI-RAW/PU200_110X_mcRun4_realistic_v3-v2/20000/087AA768-91E6-124F-B226-DC00C45D967D.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/0036F7A2-BADA-1E4E-8FE7-ABE1A9AEC350.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/007C3CAA-5209-3B47-8755-4C6D0A3A5CD2.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/00AECAEC-8DFE-8D49-AF78-A55FCEBB46B7.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/00B04974-FAC3-5A4E-B5AE-9483D8FAD5B1.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/00D64490-55F9-7E4E-B3CE-BE668F1A5938.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/PU200_pilot_110X_mcRun4_realistic_v3-v2/10000/01708416-15F1-5B47-A8A0-B32D355622DB.root'
     ) ),
    secondaryFileNames = cms.untracked.vstring()
#                            skipEvents=cms.untracked.uint32(36)
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
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
    annotation = cms.untracked.string('step1 nevts:2'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('FEVTDEBUGHLT'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('reprocess.root'),
    outputCommands = cms.untracked.vstring(
        "drop *_*_*_*",
        "keep *_l1tGMTMuons_*_*",
        "keep *_l1tGMTStubs_*_*",
        "keep *_genParticles_*_*",
        "keep *_l1tTTTracksFromTrackletEmulation_Level1TTTracks_*",
        "keep *_l1tTkMuons_*_*"
    ),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '111X_mcRun4_realistic_T15_v3', '')

##
#Calibrate Digis
process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.CalibratedDigis.dtDigiTag = "simMuonDTDigis" 
process.CalibratedDigis.scenario = 0

#DTTriggerPhase2
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.dtTriggerPhase2PrimitiveDigis.debug = False
process.dtTriggerPhase2PrimitiveDigis.dump = False
process.dtTriggerPhase2PrimitiveDigis.scenario = 0

process.load("L1Trigger.Phase2L1GMT.gmt_cff")

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.testpath=cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis*process.phase2GMT)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1TrackTrigger_step,process.testpath,process.endjob_step,process.FEVTDEBUGHLToutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
#process.options.numberOfThreads=cms.untracked.uint32(1)
#process.options.numberOfStreams=cms.untracked.uint32(0)
#process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.aging
from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000 

#call to customisation function customise_aging_1000 imported from SLHCUpgradeSimulations.Configuration.aging
#process = customise_aging_1000(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customisePhase2TTNoMC
from L1Trigger.Configuration.customisePhase2TTNoMC import customisePhase2TTNoMC 

#call to customisation function customisePhase2TTNoMC imported from L1Trigger.Configuration.customisePhase2TTNoMC
process = customisePhase2TTNoMC(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
