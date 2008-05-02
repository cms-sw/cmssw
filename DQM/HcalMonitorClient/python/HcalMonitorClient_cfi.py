import FWCore.ParameterSet.Config as cms

hcalClient = cms.EDFilter("HcalMonitorClient",
    # Are we running as standalone process?
    runningStandalone = cms.untracked.bool(False),
    CapIdMEAN_ErrThresh = cms.untracked.double(1.5),
    #Determines connection to back-end daemon
    MonitorDaemon = cms.untracked.bool(True),
    #Flags for TPG monitor
    TrigPrimClient = cms.untracked.bool(True),
    plotPedRAW = cms.untracked.bool(False),
    resetFreqEvents = cms.untracked.int32(-1),
    DeadCellClient = cms.untracked.bool(True),
    # Do you want this program to exit at the end of a run?
    enableExit = cms.untracked.bool(False),
    resetFreqLS = cms.untracked.int32(-1),
    PedestalMean_ErrThresh = cms.untracked.double(2.0),
    # Analysis jobs to be run
    DataFormatClient = cms.untracked.bool(True),
    CapIdRMS_ErrThresh = cms.untracked.double(0.25),
    HotCellClient = cms.untracked.bool(True),
    processName = cms.untracked.string(''),
    LEDRMS_ErrThresh = cms.untracked.double(0.8),
    DigiClient = cms.untracked.bool(True),
    # Operate every N updates
    diagnosticPrescaleUpdate = cms.untracked.int32(-1),
    # Name of input DQM histogram file
    inputFile = cms.untracked.string(''),
    LEDClient = cms.untracked.bool(True),
    resetFreqTime = cms.untracked.int32(-1),
    PedestalClient = cms.untracked.bool(True),
    LEDMEAN_ErrThresh = cms.untracked.double(2.25),
    # Operate every N lumi sections
    diagnosticPrescaleLS = cms.untracked.int32(-1),
    #Which subdetectors are present?
    subDetsOn = cms.untracked.vstring('HB', 
        'HE', 
        'HF', 
        'HO'),
    RecHitClient = cms.untracked.bool(True),
    # for the moment, don't run CaloTowerClient unless explicitly requested
    CaloTowerClient = cms.untracked.bool(False),
    # Reset histograms on these frequencies
    resetFreqUpdates = cms.untracked.int32(-1),
    # Directory for HTML output
    # Disabled if set to ""
    baseHtmlDir = cms.untracked.string('.'),
    DoPerChanTests = cms.untracked.bool(True),
    PedestalRMS_ErrThresh = cms.untracked.double(1.0),
    # Operate every N minutes
    diagnosticPrescaleTime = cms.untracked.int32(-1),
    # Choices for prescaling your module (-1 mean no prescale)
    # Operate every N events
    diagnosticPrescaleEvt = cms.untracked.int32(200),
    # Verbosity Switch
    debug = cms.untracked.bool(False)
)


