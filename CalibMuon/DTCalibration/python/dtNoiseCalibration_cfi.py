import FWCore.ParameterSet.Config as cms

dtNoiseCalibration = cms.EDAnalyzer("DTNoiseCalibration",
    # Label to retrieve DT digis from the event
    digiLabel = cms.InputTag('muonDTDigis'),
    # Output ROOT file name 
    rootFileName = cms.untracked.string('dtNoiseCalib.root'),
    # Trigger mode
    useTimeWindow = cms.bool(True),
    # The trigger width(ns) (full window used if useTimeWindow = False)
    triggerWidth = cms.int32(2000),
    # Time window defined as tTrig - offset (TDC counts). If defaultTtrig not set reads from DB.
    #defaultTtrig = cms.int32(322),
    timeWindowOffset = cms.int32(100),
    # Noise threshold (Hz)
    maximumNoiseRate = cms.double(2000),
    # Use absolute rate per channel or subtract average rate in layer  
    useAbsoluteRate = cms.bool(False),
    # Cells with detailed histos
    cellsWithHisto = cms.vstring(
        '-1 1 3 1 2 48',
        '0 1 7 1 1 8',
        '0 1 8 2 3 56',
        '2 1 8 2 2 56',
        '2 1 8 2 2 57',
        '2 1 12 1 2 3',
        '2 1 12 1 3 2',
        '0 2 2 1 2 3',
        '-2 3 3 1 2 2',
        '1 3 3 1 4 27',
        '1 3 3 1 4 28',
        '1 3 3 1 4 29',
        '1 3 3 1 4 30'
    )
)
