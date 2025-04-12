# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --process reHLT -s L1REPACK:Full,HLT:HIon --conditions auto:run3_hlt_HIon --data --eventcontent FEVTDEBUGHLT --datatier FEVTDEBUGHLT --era Run3_pp_on_PbPb_approxSiStripClusters_2023 -n 1 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_2023_cff import Run3_pp_on_PbPb_approxSiStripClusters_2023

process = cms.Process('reHLT',Run3_pp_on_PbPb_approxSiStripClusters_2023)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1EmulatorRepack_Full_cff')
process.load('HLTrigger.Configuration.HLT_HIon_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/scratch/nandan/inputfile_for_prehlt/HIEphemeralHLTPhysics_RAW/e1f7f325-4ca8-4fea-8d27-06b33862cf11.root'),
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
    numberOfThreads = cms.untracked.uint32(30),
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

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('FEVTDEBUGHLT'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step2_L1REPACK_HLT.root'),
    compressionAlgorithm = cms.untracked.string( "LZMA" ),
    compressionLevel = cms.untracked.int32( 4 ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_HIPhysicsRawPrime4' ) ),
    outputCommands =  cms.untracked.vstring('drop *',
      'keep *_hltSiStripClusters*_*_*',
      'keep DetIds_hltSiStripRawToDigi_*_reHLT',
      'keep FEDRawDataCollection_rawPrimeDataRepacker*_*_reHLT',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_reHLT',
      'keep edmTriggerResults_*_*_reHLT'),
    splitLevel = cms.untracked.int32(0)
)
process.HLTSiStripClusterChargeCutTight = cms.PSet(  value = cms.double( 1945.0 ) )
process.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
process.hltSiStripClusterizerForRawPrime.Clusterizer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutTight'
process.ClusterShapeHitFilterESProducer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutTight'
# Additional output definition

process.streams = cms.PSet(  PhysicsHIPhysicsRawPrime4 = cms.vstring( 'HIPhysicsRawPrime4' ) )
process.datasets = cms.PSet(  HIPhysicsRawPrime4 = cms.vstring( 'HLT_HIZeroBias_HighRate_v7' ) )
process.hltDatasetHIPhysicsRawPrime = cms.EDFilter("TriggerResultsFilter",
    hltResults = cms.InputTag(""),
    l1tIgnoreMaskAndPrescale = cms.bool(False),
    l1tResults = cms.InputTag(""),
    throw = cms.bool(True),
    triggerConditions = cms.vstring(
      'HLT_HIZeroBias_HighRate_v7'
      ),
    usePathStatus = cms.bool(True)
)
# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '141X_dataRun3_Prompt_v3', '')

# Path and EndPath definitions
process.L1RePack_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.schedule = cms.Schedule(*[ process.L1RePack_step, process.HLTriggerFirstPath, process.Status_OnCPU, process.Status_OnGPU,process.HLT_HIZeroBias_HighRate_v7,process.Dataset_HIPhysicsRawPrime4,process.FEVTDEBUGHLToutput_step])
# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
#process.schedule.insert(0, process.L1RePack_step)
#process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
#from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
#associatePatAlgosToolsTask(process)



# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
