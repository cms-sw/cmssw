import FWCore.ParameterSet.Config as cms

dtTTrigCalibration = cms.EDAnalyzer("DTTTrigCalibration",
    # Label to retrieve DT digis from the event
    digiLabel = cms.untracked.string('muonDTDigis'),
    # Switch on/off the check of noisy channels
    checkNoisyChannels = cms.untracked.bool(True),
    # Module for t0 subtraction
    tTrigMode = cms.untracked.string('DTTTrigSyncT0Only'),
    # Switch on/off the subtraction of t0 from pulses
    doSubtractT0 = cms.untracked.bool(True),
    # Max number of digi per layer to reject a chamber
    maxDigiPerLayer = cms.untracked.int32(10),
    # Name of the ROOT file which will contain the time boxes
    rootFileName = cms.untracked.string('DTTimeBoxes.root'),
    # Switch on/off the DB writing
    fitAndWrite = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    # Parameter set for t0 subtraction module
    tTrigModeConfig = cms.untracked.PSet(debug = cms.untracked.bool(False)),
    # Tbox rising edge fit parameter
    sigmaTTrigFit = cms.untracked.double(5.0),
    # the kfactor to be uploaded in the ttrig DB
    kFactor = cms.untracked.double(-0.7)
)
