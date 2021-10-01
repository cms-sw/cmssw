# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: reHLT --processName reHLT -s HLT:@relval2021 --conditions 120X_mcRun3_2021_realistic_v4 --datatier GEN-SIM-DIGI-RAW -n 10 --eventcontent FEVTDEBUGHLT --geometry DB:Extended --era Run3 --customise=HLTrigger/Configuration/customizeHLTforPatatrack.customizeHLTforPatatrackTriplets --filein /store/relval/CMSSW_12_0_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_120X_mcRun3_2021_realistic_v4_JIRA_129-v1/00000/79c06ed5-929b-4a57-a4f2-1ae90e6b38c5.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('reHLT',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_12_0_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_120X_mcRun3_2021_realistic_v4_JIRA_129-v1/00000/79c06ed5-929b-4a57-a4f2-1ae90e6b38c5.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
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
    makeTriggerResults = cms.obsolete.untracked.bool,
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
    annotation = cms.untracked.string('reHLT nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('reHLT_HLT.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '120X_mcRun3_2021_realistic_v4', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforPatatrack
from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrackTriplets 

#call to customisation function customizeHLTforPatatrackTriplets imported from HLTrigger.Configuration.customizeHLTforPatatrack
process = customizeHLTforPatatrackTriplets(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

#process.load( "HLTrigger.Timer.FastTimerService_cfi" )
#if 'MessageLogger' in process.__dict__:
#    process.MessageLogger.TriggerSummaryProducerAOD = cms.untracked.PSet()
#    process.MessageLogger.L1GtTrigReport = cms.untracked.PSet()
#    process.MessageLogger.L1TGlobalSummary = cms.untracked.PSet()
#    process.MessageLogger.HLTrigReport = cms.untracked.PSet()
#    process.MessageLogger.FastReport = cms.untracked.PSet()
#    process.MessageLogger.ThroughputService = cms.untracked.PSet()
#    process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )


############################
## Configure GPU producer ##
############################

process.hltParticleFlowRecHitHBHE = cms.EDProducer( "PFHBHERechitProducerGPU",
    producers = cms.VPSet(
      cms.PSet(  src = cms.InputTag( "hltHbherecoGPU" ),
        name = cms.string( "PFHBHERecHitCreator" ),
        qualityTests = cms.VPSet(
          cms.PSet(  threshold = cms.double( 0.8 ),
            name = cms.string( "PFRecHitQTestThreshold" ),
            cuts = cms.VPSet(
              cms.PSet(  depth = cms.vint32( 1, 2, 3, 4 ),
                threshold = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
                detectorEnum = cms.int32( 1 )
              ),
              cms.PSet(  depth = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
                threshold = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
                detectorEnum = cms.int32( 2 )
              )
            )
          ),
          cms.PSet(  flags = cms.vstring( 'Standard' ),
            cleaningThresholds = cms.vdouble( 0.0 ),
            name = cms.string( "PFRecHitQTestHCALChannel" ),
            maxSeverities = cms.vint32( 11 )
          )
        )
      )
    ),
    navigator = cms.PSet(
      name = cms.string( "PFRecHitHCALDenseIdNavigator" ),
      hcalEnums = cms.vint32( 1, 2 )
    )
)


############################
## Configure CPU producer ##
############################

#process.hltParticleFlowRecHitHBHE = cms.EDProducer( "PFRecHitProducer",
#    producers = cms.VPSet(
#      cms.PSet(  src = cms.InputTag( "hltHbherecoFromGPU" ),
#        name = cms.string( "PFHBHERecHitCreator" ),
#        qualityTests = cms.VPSet(
#          cms.PSet(  threshold = cms.double( 0.8 ),
#            name = cms.string( "PFRecHitQTestThreshold" ),
#            cuts = cms.VPSet(
#              cms.PSet(  depth = cms.vint32( 1, 2, 3, 4 ),
#                threshold = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
#                detectorEnum = cms.int32( 1 )
#              ),
#              cms.PSet(  depth = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
#                threshold = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
#                detectorEnum = cms.int32( 2 )
#              )
#            )
#          ),
#          cms.PSet(  flags = cms.vstring( 'Standard' ),
#            cleaningThresholds = cms.vdouble( 0.0 ),
#            name = cms.string( "PFRecHitQTestHCALChannel" ),
#            maxSeverities = cms.vint32( 11 )
#          )
#        )
#      )
#    ),
#    navigator = cms.PSet(
#      name = cms.string( "PFRecHitHCALDenseIdNavigator" ),
#      hcalEnums = cms.vint32( 1, 2 )
#    )
#)
