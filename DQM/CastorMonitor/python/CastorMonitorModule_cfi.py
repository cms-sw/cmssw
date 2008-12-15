import FWCore.ParameterSet.Config as cms
from copy import deepcopy

castorMonitor = cms.EDFilter("CastorMonitorModule",

                           ### GLOBAL VARIABLES
                           debug = cms.untracked.int32(1), # make debug an int so that different
                                                           # values can trigger different levels of messaging

                           ### minimum Error Rate that will cause problem histograms to be filled.
                           ### It should normally be 0, or close to it?
                           # minErrorFlag = cms.untracked.double(0.05), 

                           # Turn on/off timing diagnostic info
                           showTiming          = cms.untracked.bool(False),
                           dump2database       = cms.untracked.bool(False),
                           pedestalsInFC = cms.untracked.bool(False),
                           # thresholds = cms.untracked.vdouble(15.0, 5.0, 2.0, 1.5, 1.0),
                           
 
                           DigiMonitor = cms.untracked.bool(True),
                           digiLabel = cms.InputTag("castorDigis"),
                           CastorRecHitLabel = cms.InputTag("castorreco"),
                           RecHitsPerChannel = cms.untracked.bool(True),

                           ### PEDESTAL MONITOR
                           PedestalMonitor                  = cms.untracked.bool(True),
                           PedestalsPerChannel              = cms.untracked.bool(True), 
                           PedestalsInFC                    = cms.untracked.bool(False),
                           # minErrorFlag     = cms.untracked.double(0.05),
                           # checkNevents     = cms.untracked.int32(500),
                           #### minEntriesPerPed = cms.untracked.uint32(10), ## minimum # of events needed to calculate pedestals                                       
                           #MonitorDaemon = cms.untracked.bool(True),

                           ### RECHIT MONITOR
                           RecHitMonitor = cms.untracked.bool(True),                          
                           DigisPerChannel = cms.untracked.bool(False),

                           diagnosticPrescaleTime = cms.untracked.int32(-1),
                           diagnosticPrescaleUpdate = cms.untracked.int32(-1),
                           diagnosticPrescaleLS = cms.untracked.int32(-1),
                             
                           ### LED MONITOR
                           LEDMonitor = cms.untracked.bool(True),
                           LEDPerChannel = cms.untracked.bool(True),
                           FirstSignalBin = cms.untracked.int32(0),
                           LastSignalBin = cms.untracked.int32(9),
                           LED_ADC_Thresh = cms.untracked.double(-1000.0)
                           # checkNevents = cms.untracked.int32(250),
                           
                        
                           )


def setCastorTaskValues(process):
    # If you import this function directly, you can then set all the individual subtask values to the global settings
    # (This is useful if you've changed the global value, and you want it to propagate everywhere)

    # Set minimum value needed to put an entry into Problem histograms.  (values are between 0-1)

    # Insidious python-ness:  You need to make a copy of the process.minErrorFlag, etc. variables,
    # or future changes to PedestalMonitor_minErrorFlag will also change minErrorFlag!

    ### set minimum error value
    # minErrorFlag = deepcopy(process.minErrorFlag)
    # process.PedestalMonitor_minErrorFlag = minErrorFlag
   
    
    ### set checkNevents
    # checkNevents = deepcopy(process.checkNevents)
    # process.PedestalMonitor_checkNevents = checkNevents

    # set pedestalsInFC
    pedestalsInFC = deepcopy(process.pedestalsInFC)
    process.PedestalMonitor_pedestalsInFC = pedestalsInFC
   

    return
