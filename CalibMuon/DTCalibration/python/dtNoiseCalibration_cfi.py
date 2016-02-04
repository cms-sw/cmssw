import FWCore.ParameterSet.Config as cms

dtNoiseCalibration = cms.EDAnalyzer("DTNoiseCalibration",
    # Label to retrieve DT digis from the event
    # RAW: dtunpacker DIGI: muonDTDigis
    digiLabel = cms.untracked.string('muonDTDigis'),
    # Trigger mode
    cosmicRun = cms.untracked.bool(True),
    # Database option (to set if cosmicRun=True)
    readDB = cms.untracked.bool(False),
    # The trigger width(TDC counts) (to set if cosmicRun=True and readDB=False)
    defaultTtrig = cms.untracked.int32(322),
    theOffset = cms.untracked.double(100.),
    # The trigger width(ns) (to set if cosmicRun=False)
    TriggerWidth = cms.untracked.int32(2000),
    # Output ROOT file name 
    rootFileName = cms.untracked.string('dtNoiseCalib.root'),
    # Enable debug option
    debug = cms.untracked.bool(False),
    # "Fast analysis" option
    fastAnalysis = cms.untracked.bool(True),
    # Define the wheel of interest (to set if fastAnalysis=False)
    wheel = cms.untracked.int32(0),
    # Define the sector of interest (to set if fastAnalysis=False)
    sector = cms.untracked.int32(11)
)
