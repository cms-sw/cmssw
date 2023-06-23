import FWCore.ParameterSet.Config as cms

# Customization for running on testEnablesEcalHcal
#   - Call from cmsDriver.py with: `--customise DPGAnalysis/HcalNanoAOD/customiseHcalCalib_cff.customiseHcalCalib`
def customiseHcalCalib(process):
    # Add uMNio digi (special digi identifies calib event type)
    process.load("DPGAnalysis.HcalNanoAOD.hcalUMNioTable_cff")
    process.hcalNanoTask.add(process.uMNioTable)
    process.hcalNanoDigiTask.add(process.uMNioTable)

    process.load("DPGAnalysis.HcalNanoAOD.hcalUHTRTable_cff")
    process.hcalNanoTask.add(process.uHTRTable)
    process.hcalNanoDigiTask.add(process.uHTRTable)

    # Raw data has a different name, hltHcalCalibrationRaw instead of rawDataCollector
    process.hcalDigis.InputLabel = cms.InputTag('hltHcalCalibrationRaw')
    process.uHTRTable.InputLabel = process.hcalDigis.InputLabel
    
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


    #process.raw2digi_step = cms.Path(process.hcalCalibHLTFilter + process.RawToDigi)
    #process.raw2digi_step.replace(process.hcalDigis, process.hcalCalibDigis)
    #process.hcalDigiSortedTableTask.add(process.hcalCalibDigis)
    #process.hcalDigiSortedTableSeq.add(process.hcalCalibDigis)

    #process.options.SkipEvent.append('ProductNotFound')

    #process.hcalDigiSortedTable.tagQIE11 = cms.untracked.InputTag("hcalCalibDigis")
    #process.hcalDigiSortedTable.tagQIE10 = cms.untracked.InputTag("hcalCalibDigis")
    #process.hcalDigiSortedTable.tagHO    = cms.untracked.InputTag("hcalCalibDigis")

    process.load("FWCore.MessageService.MessageLogger_cfi")
    process.MessageLogger.cout.threshold = "DEBUG"
    process.MessageLogger.cerr.threshold = "DEBUG"
    # enable LogDebug messages only for specific modules
    process.MessageLogger.debugModules = ["*"]

    return process
