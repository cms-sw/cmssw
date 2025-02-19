import FWCore.ParameterSet.Config as cms

from CalibMuon.DTCalibration.dtSegmentSelection_cfi import dtSegmentSelection

dtVDriftMeanTimerCalibration = cms.EDAnalyzer("DTVDriftCalibration",
    # Segment selection
    dtSegmentSelection, 
    recHits4DLabel = cms.InputTag('dt4DSegments'), 
    rootFileName = cms.untracked.string('DTTMaxHistos.root'),
    debug = cms.untracked.bool(False),
    # Choose the chamber you want to calibrate (default = "All"), specify the chosen chamber
    # in the format "wheel station sector" (i.e. "-1 3 10")
    calibChamber = cms.untracked.string('All'),
    # Chosen granularity (N.B. bySL is the only one implemented at the moment)  
    tMaxGranularity = cms.untracked.string('bySL'),
    # The module to be used for ttrig synchronization and its set parameter
    tTrigMode = cms.string('DTTTrigSyncFromDB'),
    tTrigModeConfig = cms.PSet(
        # The velocity of signal propagation along the wire (cm/ns)
        vPropWire = cms.double(24.4),
        # Switch on/off the TOF correction for particles
        doTOFCorrection = cms.bool(True),
        tofCorrType = cms.int32(0),
        wirePropCorrType = cms.int32(0),
        # Switch on/off the correction for the signal propagation along the wire
        doWirePropCorrection = cms.bool(True),
        # Switch on/off the TO correction from pulses
        doT0Correction = cms.bool(True),
        debug = cms.untracked.bool(False),
        tTrigLabel = cms.string('')
    ),
    # Choose to calculate vDrift and t0 or just fill the TMax histograms
    findVDriftAndT0 = cms.untracked.bool(False),
    # Parameter set for DTCalibrationMap constructor
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(6),
        calibConstGranularity = cms.untracked.string('bySL'),
        calibConstFileName = cms.untracked.string('vDriftAndReso.txt')
    ),
    # Name of the txt file which will contain the calibrated v_drift
    vDriftFileName = cms.untracked.string('vDriftFromMtime.txt')
)
