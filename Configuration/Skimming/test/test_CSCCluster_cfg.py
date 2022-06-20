# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:EXOCSCCluster --dasquery=file dataset=/RelValQCD_Pt_1800_2400_14/CMSSW_12_3_0_pre6-123X_mcRun3_2021_realistic_v11-v2/GEN-SIM-RECO -n 10000 --conditions 120X_mcRun3_2021_realistic_v6 --python_filename=EXOCSCCluster_SKIM.py --processName=SKIMEXOCSCCluster --era Run3 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('SKIMEXOCSCCluster',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Skims_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_12_3_0_pre6/RelValQCD_Pt_1800_2400_14/GEN-SIM-RECO/123X_mcRun3_2021_realistic_v11-v2/10000/649f3446-1698-4910-b879-9a6d94d62a9b.root',
        '/store/relval/CMSSW_12_3_0_pre6/RelValQCD_Pt_1800_2400_14/GEN-SIM-RECO/123X_mcRun3_2021_realistic_v11-v2/10000/aa515779-dad9-4994-b7d2-d672a7c8938a.root',
        '/store/relval/CMSSW_12_3_0_pre6/RelValQCD_Pt_1800_2400_14/GEN-SIM-RECO/123X_mcRun3_2021_realistic_v11-v2/10000/b7315dce-732e-4ec7-b137-ee3bafff1cd6.root',
        '/store/relval/CMSSW_12_3_0_pre6/RelValQCD_Pt_1800_2400_14/GEN-SIM-RECO/123X_mcRun3_2021_realistic_v11-v2/10000/df8bd2c1-1e45-479d-9287-2d40fecc25d8.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
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
    annotation = cms.untracked.string('skims nevts:10000'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('skims_SKIM.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
from Configuration.Skimming.Skims_PDWG_cff  import SKIMStreamEXOCSCCluster

process.SKIMStreamEXOCSCCluster = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXOCSCClusterPath')
    ),
    dataset = cms.untracked.PSet(
        dataTier = SKIMStreamEXOCSCCluster.dataTier,
        filterName = cms.untracked.string(SKIMStreamEXOCSCCluster.name)
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('EXOCSCCluster.root'),
    outputCommands = SKIMStreamEXOCSCCluster.content
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '120X_mcRun3_2021_realistic_v6', '')

# Path and EndPath definitions
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.SKIMStreamEXOCSCClusterOutPath = cms.EndPath(process.SKIMStreamEXOCSCCluster)

# Schedule definition
process.schedule = cms.Schedule(process.EXOCSCClusterPath,process.RECOSIMoutput_step,process.SKIMStreamEXOCSCClusterOutPath)
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
