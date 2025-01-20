# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: reHLT --processName reHLT -s HLT:@relval2021 --conditions auto:phase1_2021_realistic --datatier GEN-SIM-DIGI-RAW -n 5 --eventcontent FEVTDEBUGHLT --geometry DB:Extended --era Run3 --customise=HLTrigger/Configuration/customizeHLTforPatatrack.customizeHLTforPatatrack --filein /store/relval/CMSSW_12_3_0_pre5/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v6-v1/10000/2639d8f2-aaa6-4a78-b7c2-9100a6717e6c.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('rereHLT',Run3)

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
_seedingThresholdsHB = cms.vdouble(1.0, 1.0, 1.0, 1.0)
_seedingThresholdsHE = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1)
_seedingThresholdsHBphase1 = cms.vdouble(0.125, 0.25, 0.35, 0.35)
_seedingThresholdsHEphase1 = cms.vdouble(0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275)
#updated HB RecHit threshold for 2023
_thresholdsHBphase1_2023 = cms.vdouble(0.4, 0.3, 0.3, 0.3)
#updated HB seeding threshold for 2023
_seedingThresholdsHBphase1_2023 = cms.vdouble(0.6, 0.5, 0.5, 0.5)

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

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(1),
    #input = cms.untracked.int32(100),
    input = cms.untracked.int32(1000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
# Need to use a file that contains HCAL/ECAL hits. Verify using:
# root root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_13_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_130X_mcRun3_2022_realistic_v2_HS-v4/2590000/0088b51b-0cda-40f2-95fc-590f446624ee.root -e 'Events->Print()' -q | grep -E "hltHbhereco|hltEcalRecHit"
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_130X_mcRun3_2022_realistic_v2_HS-v4/2590000/0088b51b-0cda-40f2-95fc-590f446624ee.root'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_0_8/RelValQCD_FlatPt_15_3000HS_14/GEN-SIM-DIGI-RAW/130X_mcRun3_2022_realistic_v3_2022-v1/2580000/0e63ba30-251b-4034-93ca-4d400aaa399e.root'),
    secondaryFileNames = cms.untracked.vstring(),
    #skipEvents = cms.untracked.uint32(999)
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
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
    annotation = cms.untracked.string('reHLT nevts:5'),
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

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
process = customizeHLTforMC(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

process.load( "HLTrigger.Timer.FastTimerService_cfi" )
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.TriggerSummaryProducerAOD = cms.untracked.PSet()
    process.MessageLogger.L1GtTrigReport = cms.untracked.PSet()
    process.MessageLogger.L1TGlobalSummary = cms.untracked.PSet()
    process.MessageLogger.HLTrigReport = cms.untracked.PSet()
    process.MessageLogger.FastReport = cms.untracked.PSet()
    process.MessageLogger.ThroughputService = cms.untracked.PSet()
    process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )


#####################################
##   Read command-line arguments   ##
#####################################
import sys
import argparse
parser = argparse.ArgumentParser(prog=f"{sys.argv[0]}", description='Test and validation of PFRecHitProducer with Alpaka')
parser.add_argument('-c', '--cal', type=str, default='HCAL',
                    help='Calorimeter type. Possible options: HCAL, ECAL. Default: HCAL')
parser.add_argument('-b', '--backend', type=str, default='auto',
                    help='Alpaka backend. Possible options: CPU, GPU, auto. Default: auto')
parser.add_argument('-s', '--synchronise', action='store_true', default=False,
                    help='Put synchronisation point at the end of Alpaka modules (for benchmarking performance)')
parser.add_argument('-t', '--threads', type=int, default=8,
                    help='Number of threads. Default: 8')
parser.add_argument('-d', '--debug', type=int, default=0, const=1, nargs="?",
                    help='Dump PFRecHits for first event (n>0) or first error (n<0). This applies to the n-th validation (1: Legacy vs Alpaka, 2: Legacy vs Legacy-from-Alpaka, 3: Alpaka vs Legacy-from-Alpaka). Default: 0')
args = parser.parse_args()

if(args.debug and args.threads != 1):
    args.threads = 1
    print("Number of threads set to 1 for debugging")

assert args.cal.lower() in ["hcal", "ecal", "h", "e"], "Invalid calorimeter type"
hcal = args.cal.lower() in ["hcal", "h"]
CAL = "HCAL" if hcal else "ECAL"

alpaka_backends = {
    "cpu": "alpaka_serial_sync::%s",  # Execute on CPU
    "gpu": "alpaka_cuda_async::%s",   # Execute using CUDA
    "cuda": "alpaka_cuda_async::%s",  # Execute using CUDA
    "auto": "%s@alpaka"               # Let framework choose
}
assert args.backend.lower() in alpaka_backends, "Invalid backend"
alpaka_backend_str = alpaka_backends[args.backend.lower()]



########################################
##    Legacy HBHE PFRecHit producer   ##
########################################
process.hltParticleFlowRecHitHBHE = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        hcalEnums = cms.vint32(1, 2),
        name = cms.string('PFRecHitHCALDenseIdNavigator')
    ),
    producers = cms.VPSet(cms.PSet(
        name = cms.string('PFHBHERecHitCreator'),
        qualityTests = cms.VPSet(
            cms.PSet(
                usePFThresholdsFromDB = cms.bool(True),
                cuts = cms.VPSet(
                    cms.PSet(
                        depth = cms.vint32(1, 2, 3, 4),
                        detectorEnum = cms.int32(1),
                        threshold = cms.vdouble(0.4, 0.3, 0.3, 0.3)
                    ),
                    cms.PSet(
                        depth = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                        detectorEnum = cms.int32(2),
                        threshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
                    )
                ),
                name = cms.string('PFRecHitQTestHCALThresholdVsDepth')
            ),
            cms.PSet(
                cleaningThresholds = cms.vdouble(0.0),
                flags = cms.vstring('Standard'),
                maxSeverities = cms.vint32(11),
                name = cms.string('PFRecHitQTestHCALChannel')
            )
        ),
        src = cms.InputTag("hltHbherecoLegacy")
    ))
)


#####################################
##    Legacy PFRecHit producer     ##
#####################################
if hcal:
    process.hltParticleFlowRecHit = cms.EDProducer("PFRecHitProducer",
        navigator = cms.PSet(
            hcalEnums = cms.vint32(1, 2),
            name = cms.string('PFRecHitHCALDenseIdNavigator')
        ),
        producers = cms.VPSet(cms.PSet(
            name = cms.string('PFHBHERecHitCreator'),
            qualityTests = cms.VPSet(
                cms.PSet(
                    usePFThresholdsFromDB = cms.bool(True),
                    cuts = cms.VPSet(
                        cms.PSet(
                            depth = cms.vint32(1, 2, 3, 4),
                            detectorEnum = cms.int32(1),
                            threshold = cms.vdouble(0.4, 0.3, 0.3, 0.3)
                        ),
                        cms.PSet(
                            depth = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                            detectorEnum = cms.int32(2),
                            threshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
                        )
                    ),
                    name = cms.string('PFRecHitQTestHCALThresholdVsDepth')
                ),
                cms.PSet(
                    cleaningThresholds = cms.vdouble(0.0),
                    flags = cms.vstring('Standard'),
                    maxSeverities = cms.vint32(11),
                    name = cms.string('PFRecHitQTestHCALChannel')
                )
            ),
            src = cms.InputTag("hltHbherecoLegacy")
        ))
    )
else:  # ecal
    qualityTestsECAL = cms.VPSet(
        cms.PSet(
            name = cms.string("PFRecHitQTestDBThreshold"),
            applySelectionsToAllCrystals=cms.bool(True),
        ),
        cms.PSet(
            name = cms.string("PFRecHitQTestECAL"),
            cleaningThreshold = cms.double(2.0),
            timingCleaning = cms.bool(True),
            topologicalCleaning = cms.bool(True),
            skipTTRecoveredHits = cms.bool(True)
        )
    )
    process.hltParticleFlowRecHit = cms.EDProducer("PFRecHitProducer",
        navigator = cms.PSet(
            name = cms.string("PFRecHitECALNavigator"),
            barrel = cms.PSet( ),
            endcap = cms.PSet( )
        ),
        producers = cms.VPSet(
            cms.PSet(
                name = cms.string("PFEBRecHitCreator"),
                src  = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
                srFlags = cms.InputTag(""),
                qualityTests = qualityTestsECAL
            ),
            cms.PSet(
                name = cms.string("PFEERecHitCreator"),
                src  = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
                srFlags = cms.InputTag(""),
                qualityTests = qualityTestsECAL
            )
        )
    )


#####################################
##    Alpaka PFRecHit producer     ##
#####################################
# Convert legacy CaloRecHits to CaloRecHitSoA
if hcal:
    process.hltParticleFlowRecHitToSoA = cms.EDProducer(alpaka_backend_str % "HCALRecHitSoAProducer",
        src = cms.InputTag("hltHbherecoLegacy"),
        synchronise = cms.untracked.bool(args.synchronise)
    )
else:  # ecal
    process.hltParticleFlowRecHitEBToSoA = cms.EDProducer(alpaka_backend_str % "ECALRecHitSoAProducer",
        src = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
        synchronise = cms.untracked.bool(args.synchronise)
    )
    process.hltParticleFlowRecHitEEToSoA = cms.EDProducer(alpaka_backend_str % "ECALRecHitSoAProducer",
        src = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
        synchronise = cms.untracked.bool(args.synchronise)
    )

# Construct topology and cut parameter information
process.pfRecHitTopologyRecordSource = cms.ESSource('EmptyESSource',
    recordName = cms.string(f'PFRecHit{CAL}TopologyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.pfRecHitParamsRecordSource = cms.ESSource('EmptyESSource',
    recordName = cms.string(f'PFRecHit{CAL}ParamsRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.hltParticleFlowRecHitTopologyESProducer = cms.ESProducer(alpaka_backend_str % f"PFRecHit{CAL}TopologyESProducer",
        usePFThresholdsFromDB = cms.bool(True))
if hcal:
    process.hltParticleFlowRecHitParamsESProducer = cms.ESProducer(alpaka_backend_str % "PFRecHitHCALParamsESProducer",
        energyThresholdsHB = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
        energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
    )
else:  # ecal
    process.hltParticleFlowRecHitParamsESProducer = cms.ESProducer(alpaka_backend_str % "PFRecHitECALParamsESProducer")

# Construct PFRecHitSoA
if hcal:
    process.hltParticleFlowPFRecHitAlpaka = cms.EDProducer(alpaka_backend_str % "PFRecHitSoAProducerHCAL",
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hltParticleFlowRecHitToSoA"),
                params = cms.ESInputTag("hltParticleFlowRecHitParamsESProducer:"),
            )
        ),
        topology = cms.ESInputTag("hltParticleFlowRecHitTopologyESProducer:"),
        synchronise = cms.untracked.bool(args.synchronise)
    )
else:  # ecal
    process.hltParticleFlowPFRecHitAlpaka = cms.EDProducer(alpaka_backend_str % "PFRecHitSoAProducerECAL",
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hltParticleFlowRecHitEBToSoA"),
                params = cms.ESInputTag("hltParticleFlowRecHitParamsESProducer:")
            ),
            cms.PSet(
                src = cms.InputTag("hltParticleFlowRecHitEEToSoA"),
                params = cms.ESInputTag("hltParticleFlowRecHitParamsESProducer:")
            )
        ),
        topology = cms.ESInputTag("hltParticleFlowRecHitTopologyESProducer:"),
        synchronise = cms.bool(args.synchronise)
    )

# Convert Alpaka PFRecHits to legacy format (for validation)
process.hltParticleFlowAlpakaToLegacyPFRecHits = cms.EDProducer("LegacyPFRecHitProducer",
    src = cms.InputTag("hltParticleFlowPFRecHitAlpaka")
)


#####################################
##       PFRecHit validation       ##
#####################################
# Validate legacy format from legacy module vs SoA format from Alpaka module
# This is the main Alpaka vs legacy test
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.hltParticleFlowPFRecHitComparison = DQMEDAnalyzer("PFRecHitProducerTest",
    #caloRecHits = cms.untracked.InputTag("hltParticleFlowRecHitToSoA"),
    pfRecHitsSource1 = cms.untracked.InputTag("hltParticleFlowRecHit"),
    pfRecHitsSource2 = cms.untracked.InputTag("hltParticleFlowPFRecHitAlpaka"),
    pfRecHitsType1 = cms.untracked.string("legacy"),
    pfRecHitsType2 = cms.untracked.string("alpaka"),
    title = cms.untracked.string("Legacy vs Alpaka"),
    dumpFirstEvent = cms.untracked.bool(args.debug == 1),
    dumpFirstError = cms.untracked.bool(args.debug == -1),
    strictCompare = cms.untracked.bool(True)
)

# Validate legacy format from legacy module vs legacy format from Alpaka module
process.hltParticleFlowAlpakaToLegacyPFRecHitsComparison1 = DQMEDAnalyzer("PFRecHitProducerTest",
    pfRecHitsSource1 = cms.untracked.InputTag("hltParticleFlowRecHitHBHE"),
    pfRecHitsSource2 = cms.untracked.InputTag("hltParticleFlowAlpakaToLegacyPFRecHits"),
    pfRecHitsType1 = cms.untracked.string("legacy"),
    pfRecHitsType2 = cms.untracked.string("legacy"),
    title = cms.untracked.string("Legacy vs Legacy-from-Alpaka"),
    dumpFirstEvent = cms.untracked.bool(args.debug == 2),
    dumpFirstError = cms.untracked.bool(args.debug == -2),
    strictCompare = cms.untracked.bool(True)
)

# Validate SoA format from Alpaka module vs legacy format from Alpaka module
# This tests the SoA-to-legacy conversion module
process.hltParticleFlowAlpakaToLegacyPFRecHitsComparison2 = DQMEDAnalyzer("PFRecHitProducerTest",
    pfRecHitsSource1 = cms.untracked.InputTag("hltParticleFlowPFRecHitAlpaka"),
    pfRecHitsSource2 = cms.untracked.InputTag("hltParticleFlowAlpakaToLegacyPFRecHits"),
    pfRecHitsType1 = cms.untracked.string("alpaka"),
    pfRecHitsType2 = cms.untracked.string("legacy"),
    title = cms.untracked.string("Alpaka vs Legacy-from-Alpaka"),
    dumpFirstEvent = cms.untracked.bool(args.debug == 3),
    dumpFirstError = cms.untracked.bool(args.debug == -3),
    strictCompare = cms.untracked.bool(True)
)

#Move Onto Clustering
import RecoParticleFlow.PFClusterProducer.modules as _modules
process.hltParticleFlowPFClusterAlpaka = getattr(_modules, (alpaka_backend_str % "PFClusterSoAProducer").replace("::", "_").replace("@", "_"))(
    pfRecHits = "hltParticleFlowPFRecHitAlpaka",
    topology = "hltParticleFlowRecHitTopologyESProducer:",
    pfClusterBuilder = dict(maxIterations = 5),
    synchronise = args.synchronise
)
for idx, x in enumerate(process.hltParticleFlowPFClusterAlpaka.initialClusteringStep.thresholdsByDetector):
    for idy, y in enumerate(process.hltParticleFlowClusterHBHE.initialClusteringStep.thresholdsByDetector):
        if x.detector == y.detector:
            x.gatheringThreshold = y.gatheringThreshold
for idx, x in enumerate(process.hltParticleFlowPFClusterAlpaka.pfClusterBuilder.recHitEnergyNorms):
    for idy, y in enumerate(process.hltParticleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms):
        if x.detector == y.detector:
            x.recHitEnergyNorm = y.recHitEnergyNorm
for idx, x in enumerate(process.hltParticleFlowPFClusterAlpaka.seedFinder.thresholdsByDetector):
    for idy, y in enumerate(process.hltParticleFlowClusterHBHE.seedFinder.thresholdsByDetector):
        if x.detector == y.detector:
            x.seedingThreshold = y.seedingThreshold


# Create legacy PFClusters

process.hltParticleFlowAlpakaToLegacyPFClusters = cms.EDProducer("LegacyPFClusterProducer",
                                                                 src = cms.InputTag("hltParticleFlowPFClusterAlpaka"),
                                                                 pfClusterBuilder = process.hltParticleFlowClusterHBHE.pfClusterBuilder,
                                                                 usePFThresholdsFromDB = cms.bool(True),
                                                                 recHitsSource = cms.InputTag("hltParticleFlowAlpakaToLegacyPFRecHits"))
process.hltParticleFlowAlpakaToLegacyPFClusters.PFRecHitsLabelIn = cms.InputTag("hltParticleFlowPFRecHitAlpaka")

process.hltParticleFlowClusterHBHE.pfClusterBuilder.maxIterations = 5
process.hltParticleFlowClusterHBHE.usePFThresholdsFromDB = cms.bool(True)

# Additional customization
process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring('drop  *_*_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltParticleFlowRecHitToSoA_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltParticleFlowPFRecHitAlpaka_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltParticleFlowAlpakaToLegacyPFClusters_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltParticleFlowClusterHBHE_*_*')

# Path/sequence definitions
path = process.hltHcalDigis
path += process.hltHbherecoLegacy
path += process.hltParticleFlowRecHit               # Construct PFRecHits on CPU
if hcal:
    path += process.hltParticleFlowRecHitToSoA     # Convert legacy calorimeter hits to SoA (HCAL barrel+endcap)
else:  # ecal
    path += process.hltParticleFlowRecHitEBToSoA   # Convert legacy calorimeter hits to SoA (ECAL barrel)
    path += process.hltParticleFlowRecHitEEToSoA   # Convert legacy calorimeter hits to SoA (ECAL endcap)
path += process.hltParticleFlowPFRecHitAlpaka      # Construct PFRecHits SoA
path += process.hltParticleFlowRecHitHBHE               # Construct Legacy PFRecHits
path += process.hltParticleFlowClusterHBHE
path += process.hltParticleFlowPFRecHitComparison  # Validate Alpaka vs CPU
path += process.hltParticleFlowAlpakaToLegacyPFRecHits             # Convert Alpaka PFRecHits SoA to legacy format
path += process.hltParticleFlowAlpakaToLegacyPFRecHitsComparison1  # Validate legacy-format-from-alpaka vs regular legacy format
path += process.hltParticleFlowAlpakaToLegacyPFRecHitsComparison2  # Validate Alpaka format vs legacy-format-from-alpaka

path += process.hltParticleFlowPFClusterAlpaka
path += process.hltParticleFlowAlpakaToLegacyPFClusters

process.PFClusterAlpakaValidationTask = cms.EndPath(path)
process.schedule = cms.Schedule(process.PFClusterAlpakaValidationTask)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
process.options.numberOfThreads = cms.untracked.uint32(args.threads)

# Save DQM output
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:DQMIO.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)
process.DQMTask = cms.EndPath(process.DQMoutput)
process.schedule.append(process.DQMTask)
