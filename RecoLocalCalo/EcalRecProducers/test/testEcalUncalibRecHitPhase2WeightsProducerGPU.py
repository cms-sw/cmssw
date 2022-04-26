import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal

process = cms.Process('RECO',Phase2C10,phase2_ecal_devel, gpu, gpuValidationEcal)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D60Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2_cff')
process.load('RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.load('HLTrigger.Timer.FastTimerService_cfi')
process.FastTimerService.enableDQM = False
process.FastTimerService.printRunSummary = False
process.FastTimerService.printJobSummary = True
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = 'resources.json'
process.MessageLogger.FastReport = cms.untracked.PSet()



# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/group/dpg_ecal/comm_ecal/upgrade/Phase2CMSSW///RelValTTbar_14TeV_ecaldigi_123X_mcRun4_realistic_v3_2026D77noPU-v1.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
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
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3GPU nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Set up the DQM GPU validation task
process.ecalMonitorTaskEcalOnly.workers = ["GpuTask"]
process.ecalMonitorTaskEcalOnly.collectionTags.EBCpuUncalibRecHit = "ecalUncalibRecHitPhase2@cpu:EcalUncalibRecHitsEB"
process.ecalMonitorTaskEcalOnly.collectionTags.EBGpuUncalibRecHit = "ecalUncalibRecHitPhase2@cuda:EcalUncalibRecHitsEB"

# Output definition
outputCommand = process.FEVTDEBUGHLTEventContent.outputCommands
outputCommand.append('keep *_ecalUncalibRecHitPhase2*_*_RECO')
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3GPU.root'),
    outputCommands = outputCommand,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3GPU_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# Path and EndPath definitions
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(cms.Sequence(cms.Task(
    process.bunchSpacingProducer,
    process.ecalUncalibRecHitPhase2Task
)))
process.dqmoffline_step = cms.EndPath(process.DQMOfflineEcalOnly)

process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)



# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion