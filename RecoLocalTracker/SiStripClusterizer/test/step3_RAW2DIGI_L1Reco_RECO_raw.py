# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:run3_data_HIon -s RAW2DIGI,L1Reco,RECO --datatier RECO --eventcontent RECO --data --process reRECO --scenario pp --era Run3_pp_on_PbPb_approxSiStripClusters_2023 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 --hltProcess reHLT -n 100 --no_exec --repacked
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_2025_cff import Run3_pp_on_PbPb_approxSiStripClusters_2025

process = cms.Process('reRECO',Run3_pp_on_PbPb_approxSiStripClusters_2025)
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_DataMapper_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
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
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step3_RAW2DIGI_L1Reco_RECO_raw.root'),
    outputCommands = cms.untracked.vstring( 'drop *',
'keep *_*offlinePrimaryVertices_*_*',
'keep *_*siStripClusters*_*_*',
'keep *_*generalTracks*_*_*',
'keep *_hltSiStripClusters2ApproxClusters_*_*',
'keep *_ak4*Jets_*_*',
'keep *_*pfMet*_*_*'),
    splitLevel = cms.untracked.int32(0)
)

process.siStripClusters.Clusterizer.clusterChargeCut.refToPSet_ = cms.string('SiStripClusterChargeCutTight')
# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '141X_dataRun3_Prompt_v3', '')
process.siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
    Algorithms = cms.PSet(
        APVInspectMode = cms.string('Hybrid'),
        APVRestoreMode = cms.string('BaselineFollower'),
        ApplyBaselineCleaner = cms.bool(True),
        ApplyBaselineRejection = cms.bool(True),
        CleaningSequence = cms.uint32(1),
        CommonModeNoiseSubtractionMode = cms.string('IteratedMedian'),
        CutToAvoidSignal = cms.double(2.0),
        DeltaCMThreshold = cms.uint32(20),
        Deviation = cms.uint32(25),
        ForceNoRestore = cms.bool(False),
        Fraction = cms.double(0.2),
        Iterations = cms.int32(3),
        MeanCM = cms.int32(0),
        PedestalSubtractionFedMode = cms.bool(False),
        SiStripFedZeroSuppressionMode = cms.uint32(4),
        TruncateInSuppressor = cms.bool(True),
        Use10bitsTruncation = cms.bool(False),
        consecThreshold = cms.uint32(5),
        discontinuityThreshold = cms.int32(12),
        distortionThreshold = cms.uint32(20),
        doAPVRestore = cms.bool(True),
        filteredBaselineDerivativeSumSquare = cms.double(30.0),
        filteredBaselineMax = cms.double(6.0),
        hitStripThreshold = cms.uint32(40),
        lastGradient = cms.int32(10),
        minStripsToFit = cms.uint32(4),
        nSaturatedStrip = cms.uint32(2),
        nSigmaNoiseDerTh = cms.uint32(4),
        nSmooth = cms.uint32(9),
        restoreThreshold = cms.double(0.5),
        sizeWindow = cms.int32(1),
        slopeX = cms.int32(3),
        slopeY = cms.int32(4),
        useCMMeanMap = cms.bool(False),
        useRealMeanCM = cms.bool(False),
        widthCluster = cms.int32(64)
    ),
    RawDigiProducersList = cms.VInputTag("siStripDigis:VirginRaw", "siStripDigis:ProcessedRaw", "siStripDigis:ScopeMode", "siStripDigis:ZeroSuppressed"),
    fixCM = cms.bool(False),
    produceBaselinePoints = cms.bool(False),
    produceCalculatedBaseline = cms.bool(False),
    produceHybridFormat = cms.bool(False),
    produceRawDigis = cms.bool(False),
    storeCM = cms.bool(False),
    storeInZScollBadAPV = cms.bool(True)
)

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
process.raw2digi_step = cms.Path(process.RawToDigi+process.siStripZeroSuppression)
process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed"),cms.InputTag("siStripDigis","VirginRaw"), cms.InputTag("siStripDigis","ProcessedRaw"), cms.InputTag("siStripDigis","ScopeMode"))
process.siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag("siStripZeroSuppression","ZeroSuppressed"), cms.InputTag("siStripZeroSuppression","VirginRaw"), cms.InputTag("siStripZeroSuppression","ProcessedRaw"), cms.InputTag("siStripZeroSuppression","ScopeMode"))
#process.zero_suppresed = cms.Task(process.siStripZeroSuppression)
#process.RawToDigiTask = cms.Task(process.RawToDigiTask,process.zero_suppresed)
#process.raw2digi_step = cms.Path(cms.Task(process.RawToDigiTask, process.zero_suppresed))
process.striptrackerlocalrecoTask = cms.Task(process.siStripClusters, process.siStripMatchedRecHits)
# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RECOoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.RecoTLR
from Configuration.DataProcessing.RecoTLR import customisePostEra_Run3 

#call to customisation function customisePostEra_Run3 imported from Configuration.DataProcessing.RecoTLR
process = customisePostEra_Run3(process)

# End of customisation functions


# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
