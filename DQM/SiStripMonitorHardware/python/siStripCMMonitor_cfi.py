import FWCore.ParameterSet.Config as cms

siStripCMMonitor = cms.EDAnalyzer(
    "SiStripCMMonitorPlugin",
    #Raw data collection
    RawDataTag = cms.InputTag('source'),
    #Folder in DQM Store to write global histograms to
    HistogramFolderName = cms.string('SiStrip/ReadoutView/FedMedians'),
    #ids of FEDs which will have detailed histograms made
    FedIdVec = cms.vuint32(),
    #Fill all detailed histograms at FED level even if they will be empty (so that files can be merged)
    FillAllDetailedHistograms = cms.bool(False),
    #do histos vs time with time=event number. Default time = orbit number (s).
    FillWithEventNumber = cms.bool(False),
    FillWithLocalEventNumber = cms.bool(False),
    #Whether to dump buffer info and raw data if any error is found: 
    #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
    PrintDebugMessages = cms.uint32(1),
    #PrintDebugMessages = cms.bool(False),
    #Whether to write the DQM store to a file at the end of the run and the file name
    WriteDQMStore = cms.bool(True),
    DQMStoreFileName = cms.string('DQMStore.root'),
    TimeHistogramConfig = cms.PSet(
        Enabled = cms.bool(True),
        NBins = cms.uint32(1000),
        Min = cms.double(0),
        Max = cms.double(10000)
        ),
    MedianAPV1vsAPV0HistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MedianAPV0minusAPV1HistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MeanCMPerFedvsFedIdHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MeanCMPerFedvsTimeHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    VariationsPerFedvsFedIdHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    VariationsPerFedvsTimeHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MedianAPV1vsAPV0perFEDHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MedianAPV0minusAPV1perFEDHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MedianperChannelHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    MedianAPV0minusAPV1perChannelHistogramConfig = cms.PSet(
        Enabled = cms.bool(True)
        ),
    #TkHistoMap
    TkHistoMapHistogramConfig = cms.PSet( 
        Enabled = cms.bool(True) 
        )

    )
