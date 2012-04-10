import FWCore.ParameterSet.Config as cms

dtT0WireCalibration = cms.EDAnalyzer("DTT0Calibration",
    # Set to true to correct t0's from chamber mean
    # Otherwise write absolute t0's (to be corrected in a second step) 
    correctByChamberMean = cms.bool(True),
    # Cells for which you want the histos (default = None)
    cellsWithHisto = cms.untracked.vstring(),
    # Label to retrieve DT digis from the event
    digiLabel = cms.untracked.string('muonDTDigis'),
    calibSector = cms.untracked.string('All'),
    # Chose the wheel, sector (default = All)
    calibWheel = cms.untracked.string('All'),
    # Number of events to be used for the t0 per layer histos
    eventsForWireT0 = cms.uint32(25000),
    # Name of the ROOT file which will contain the test pulse times per layer
    rootFileName = cms.untracked.string('DTTestPulses.root'),
    debug = cms.untracked.bool(False),
    rejectDigiFromPeak = cms.uint32(50),
    # Acceptance for TP peak width
    tpPeakWidth = cms.double(15.0),
    # Number of events to be used for the t0 per layer histos
    eventsForLayerT0 = cms.uint32(5000)
)
