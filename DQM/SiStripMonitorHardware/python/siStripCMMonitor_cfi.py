import FWCore.ParameterSet.Config as cms

siStripCMMonitor = cms.EDAnalyzer(
    "SiStripCMMonitorPlugin",
    #Raw data collection
    RawDataTag = cms.untracked.InputTag('source'),
    #Folder in DQM Store to write global histograms to
    HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedMedians'),
    #ids of FEDs which will have detailed histograms made
    FedIdVec = cms.untracked.vuint32(),
    #Fill all detailed histograms at FED level even if they will be empty (so that files can be merged)
    FillAllDetailedHistograms = cms.untracked.bool(False),
    #do histos vs time with time=event number. Default time = orbit number (s).
    FillWithEventNumber = cms.untracked.bool(False),
    FillWithLocalEventNumber = cms.untracked.bool(False),
    #Whether to dump buffer info and raw data if any error is found: 
    #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
    PrintDebugMessages = cms.untracked.uint32(1),
    #PrintDebugMessages = cms.untracked.bool(False),
    #Whether to write the DQM store to a file at the end of the run and the file name
    WriteDQMStore = cms.untracked.bool(True),
    DQMStoreFileName = cms.untracked.string('DQMStore.root'),
    TimeHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True),
        NBins = cms.untracked.uint32(1000),
        Min = cms.untracked.double(0),
        Max = cms.untracked.double(10000)
        ),
    MedianAPV1vsAPV0HistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MedianAPV0minusAPV1HistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MeanCMPerFedvsFedIdHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MeanCMPerFedvsTimeHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    VariationsPerFedvsFedIdHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    VariationsPerFedvsTimeHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MedianAPV1vsAPV0perFEDHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MedianAPV0minusAPV1perFEDHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MedianperChannelHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    MedianAPV0minusAPV1perChannelHistogramConfig = cms.untracked.PSet(
        Enabled = cms.untracked.bool(True)
        ),
    #TkHistoMap
    TkHistoMapHistogramConfig = cms.untracked.PSet( 
        Enabled = cms.untracked.bool(True) 
        )

    )
