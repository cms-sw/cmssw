import FWCore.ParameterSet.Config as cms


dtTPmonitor = cms.EDAnalyzer("DTDigiTask",
    # set the max TDC counts for the time-box (6400 or 1600)
    maxTDCCounts = cms.untracked.int32(6400),
    # bin size for the time boxes
    timeBoxGranularity = cms.untracked.int32(4),
    # Set to true to read the ttrig from the DB
    readDB = cms.untracked.bool(False),
    # Value of the ttrig pedestal used when not reading from DB
    defaultTtrig = cms.int32(3450),
    # the label to retrieve the DT digis
    dtDigiLabel = cms.InputTag("dtunpacker"),
    # check the noisy flag in the DB and use it
    checkNoisyChannels = cms.untracked.bool(True),
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    inTimeHitsLowerBound = cms.int32(50),
    inTimeHitsUpperBound = cms.int32(50),
    # switch on debug verbosity
    debug = cms.untracked.bool(False),
    # if true access LTC digis
    localrun = cms.untracked.bool(True),
    # define the boundaries for in-time hits (ns)
    defaultTmax = cms.int32(50),
    performPerWireT0Calibration = cms.bool(True),
    # the     # of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(100),
    doAllHitsOccupancies = cms.untracked.bool(False),
    doNoiseOccupancies = cms.untracked.bool(False),
    doInTimeOccupancies = cms.untracked.bool(True),                                
    # switch on the mode for running on test pulses (different top folder)
    testPulseMode = cms.untracked.bool(True),
    # switch for filtering on synch noise events (threshold on # of digis per chamber)
    filterSyncNoise = cms.untracked.bool(False),
    # threshold on # of digis per chamber to define sync noise
    maxTDCHitsPerChamber = cms.untracked.int32(50),
    # switch for time boxes with layer granularity (commissioning only)                           
    doLayerTimeBoxes = cms.untracked.bool(False)
)





