import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTPmonitor = DQMEDAnalyzer('DTDigiTask',
    # set the max TDC counts for the time-box (6400 or 1600)
    maxTTMounts = cms.untracked.int32(1600),
    # bin size for the time boxes
    timeBoxGranularity = cms.untracked.int32(4),
    # Set to true to read the ttrig from the DB
    readDB = cms.untracked.bool(False),
    # Value of the ttrig pedestal used when not reading from DB
    defaultTtrig = cms.int32(3450),
    # the label to retrieve the DT digis
    dtDigiLabel = cms.InputTag('dtunpacker'),
    # check the noisy flag in the DB and use it
    checkNoisyChannels = cms.untracked.bool(True),
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    inTimeHitsLowerBound = cms.int32(0),
    inTimeHitsUpperBound = cms.int32(0),
    # switch on debug verbosity
    debug = cms.untracked.bool(False),
    # if true access LTC digis
    localrun = cms.untracked.bool(True),
    # define the boundaries for in-time hits (ns)
    defaultTmax = cms.int32(50),
    performPerWireT0Calibration = cms.bool(True),
    # the     # of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(400),
    doAllHitsOccupancies = cms.untracked.bool(False),
    doNoiseOccupancies = cms.untracked.bool(False),
    doInTimeOccupancies = cms.untracked.bool(True),                                
    # switch on the mode for running on test pulses (different top folder)
    testPulseMode = cms.untracked.bool(True),
    # switch on the mode for running on slice test (different top folder and customizations)
    sliceTestMode = cms.untracked.bool(False),
    # time pedestal defining the lower edge of the timebox plots
    tdcPedestal = cms.untracked.int32(0),
    # switch for filtering on synch noise events (threshold on # of digis per chamber)
    filterSyncNoise = cms.untracked.bool(False),
    # threshold on # of digis per chamber to define sync noise
    maxTDCHitsPerChamber = cms.untracked.int32(100),
    # switch for time boxes with layer granularity (commissioning only)                           
    doLayerTimeBoxes = cms.untracked.bool(False)
)





