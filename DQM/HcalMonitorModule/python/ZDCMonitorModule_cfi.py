import FWCore.ParameterSet.Config as cms
from copy import deepcopy

zdcMonitor = cms.EDAnalyzer("ZDCMonitorModule",

                          # GLOBAL VARIABLES
                          debug = cms.untracked.int32(0), # make debug an int so that different values can trigger different levels of messaging
                          Online = cms.untracked.bool(False), # control online/offline differences in code
                          
                          # number of luminosity blocks to check
                          Nlumiblocks = cms.untracked.int32(1000),
                          AllowedCalibTypes = cms.untracked.vint32([0,1,2,3,4,5,6,7]),
                          BadCells = cms.untracked.vstring(),
                          
                          # Determine whether or not to check individual subdetectors
                          checkZDC= cms.untracked.bool(True),
                          checkNevents = cms.untracked.int32(1000),
                          subSystemFolder = cms.untracked.string("Hcal/ZDCMonitor"), # change to "ZDC" when code is finalized
                           
                          FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector"),
                          
                           # Turn on/off timing diagnostic info
                          showTiming			= cms.untracked.bool(False), # shows time taken by each process
                           diagnosticPrescaleLS		= cms.untracked.int32(-1),
                          diagnosticPrescaleEvt	= cms.untracked.int32(-1),
                          
                          #Specify Pedestal Units 
                          pedestalsInFC			= cms.untracked.bool(True),
                          #Specify Digis
                          digiLabel = cms.InputTag("hcalDigis"), 
                          #Specify RecHits 
                          zdcRecHitLabel = cms.InputTag("zdcreco"),
                          
                          # ZDC MONITOR
                          ZDCMonitor				= cms.untracked.bool(True),
                          ZDCMonitor_checkNevents		= cms.untracked.int32(1000),
                          ZDCMonitor_deadthresholdrate		= cms.untracked.double(0.),
                          
                           gtLabel = cms.InputTag("l1GtUnpack"),
                          
                          )

def setZDCTaskValues(process):
    # If you import this function directly, you can then set all the individual subtask values to the global settings
    # (This is useful if you've changed the global value, and you want it to propagate everywhere)

    # set checkNevents -- soon to be deprecated in favor of checking once/lumi block
    checkNevents = deepcopy(process.checkNevents.value())
    process.ZDCMonitor_checkNevents			= checkNevents
    
    # set pedestalsInFC
    pedestalsInFC = deepcopy(process.pedestalsInFC.value())
    return
