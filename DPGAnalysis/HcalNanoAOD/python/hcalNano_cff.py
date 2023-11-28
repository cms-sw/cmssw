from PhysicsTools.NanoAOD.common_cff import Var,CandVars
from DPGAnalysis.HcalNanoAOD.hcalRecHitTable_cff import *
from DPGAnalysis.HcalNanoAOD.hcalDigiSortedTable_cff import *
from DPGAnalysis.HcalNanoAOD.hcalDetIdTable_cff import *

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

hcalNanoTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalDigiSortedTableTask, 
    hcalRecHitTableTask, 
)

hcalNanoDigiTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalDigiSortedTableTask, 
)

hcalNanoRecHitTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalRecHitTableTask, 
)

# Tasks for HCAL AlCa workflows
hcalNanoPhiSymTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hbheRecHitTable,
    hfRecHitTable,
)

hcalNanoIsoTrkTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hbheRecHitTable,
)

# Customization for running on testEnablesEcalHcal
#   - Call from cmsDriver.py with: `--customise DPGAnalysis/HcalNanoAOD/hcalNano_cff.customiseHcalCalib`
def customiseHcalCalib(process):
    # Add uMNio digi (special digi identifies calib event type)
    process.load("DPGAnalysis.HcalNanoAOD.hcalUMNioTable_cff")
    process.hcalNanoTask.add(process.uMNioTable)
    process.hcalNanoDigiTask.add(process.uMNioTable)

    # Raw data has a different name, hltHcalCalibrationRaw instead of rawDataCollector
    process.hcalDigis.InputLabel = cms.InputTag('hltHcalCalibrationRaw')

    # Create EDFilter for HLT_HcalCalibration
    # (HCAL raw data is not present in ECAL-triggered events, annoyingly. The filter stops downstream modules from throwing ProductNotFound.)
    process.hcalCalibHLTFilter = cms.EDFilter("TriggerResultsFilter",
        triggerConditions = cms.vstring(
          'HLT_HcalCalibration_v* / 1'),
        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
        l1tResults = cms.InputTag( "" ),
        l1tIgnoreMask = cms.bool( False ),
        l1techIgnorePrescales = cms.bool( False ),
        daqPartitions = cms.uint32( 1 ),
        throw = cms.bool( True )
    )

    # Remove hcalDigis from normal raw2digi task, and put on a sequence after the HLT filter
    process.RawToDigiTask.remove(process.hcalDigis)
    process.hcalCalibDigiSequence = cms.Sequence(process.hcalCalibHLTFilter + process.hcalDigis)
    process.raw2digi_step = cms.Path(process.hcalCalibDigiSequence, process.RawToDigiTask)

    # Insert the HLT filter at start of user path and nanoaod endpath
    process.user_step.insert(0, process.hcalCalibHLTFilter)
    process.NANOAODoutput_step.insert(0, process.hcalCalibHLTFilter)

    return process

# Customization for running on HCAL local run data
#   - Call from cmsDriver.py with: `--customise DPGAnalysis/HcalNanoAOD/customise_hcalLocal_cff.customiseHcalLocal`
def customiseHcalLocal(process):
    input_files = process.source.fileNames
    max_events = process.maxEvents.input
    process.source = cms.Source("HcalTBSource",
        fileNames = input_files,
        maxEvents = max_events,
        firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID([]),
    )
    process.hcalDigis.InputLabel = cms.InputTag('source')

    # Uncomment if ZDC digis (QIE10, nTS=10) are causing problems
    #process.hcalDigis.saveQIE10DataNSamples = cms.untracked.vint32(10) 
    #process.hcalDigis.saveQIE10DataTags = cms.untracked.vstring("ZDC")

    if hasattr(process, "hcalDigiSortedTableTask"):
        process.hcalDigiSortedTable.nTS_HB = cms.untracked.uint32(8)
        process.hcalDigiSortedTable.nTS_HE = cms.untracked.uint32(8)
        process.hcalDigiSortedTable.nTS_HF = cms.untracked.uint32(6)
        process.hcalDigiSortedTable.nTS_HO = cms.untracked.uint32(10)

    process.load("DPGAnalysis.HcalNanoAOD.hcalUMNioTable_cff")
    if hasattr(process, "hcalNanoTask"):
        process.hcalNanoTask.add(process.uMNioTable)
    if hasattr(process, "hcalNanoDigiTask"):
        process.hcalNanoDigiTask.add(process.uMNioTable)

    return process
