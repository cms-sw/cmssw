import FWCore.ParameterSet.Config as cms

siStripFEDMonitor = cms.EDAnalyzer("SiStripFEDMonitorPlugin",
  #Raw data collection
  RawDataTag = cms.InputTag('source'),
  #Folder in DQM Store to write global histograms to
  HistogramFolderName = cms.string('SiStrip/ReadoutView/FedMonitoringSummary'),
  #Fill all detailed histograms at FED level even if they will be empty (so that files can be merged)
  FillAllDetailedHistograms = cms.bool(False),
  #do histos vs time with time=event number. Default time = orbit number (s).
  FillWithEventNumber = cms.bool(False),
  #Whether to dump buffer info and raw data if any error is found: 
  #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
  PrintDebugMessages = cms.uint32(1),
  #PrintDebugMessages = cms.bool(False),
  #Whether to write the DQM store to a file at the end of the run and the file name
  WriteDQMStore = cms.bool(False),
  DQMStoreFileName = cms.string('DQMStore.root'),
  #Histogram configuration
  #lumi histogram
  ErrorFractionByLumiBlockHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),          
  #Global/summary histograms
  DataPresentHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  AnyFEDErrorsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  AnyDAQProblemsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  AnyFEProblemsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  CorruptBuffersHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadChannelStatusBitsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadActiveChannelStatusBitsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  #Sub sets of FE problems
  FEOverflowsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FEMissingHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadMajorityAddressesHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FETimeDiffTIBHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FETimeDiffTOBHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FETimeDiffTECBHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FETimeDiffTECFHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  ApveAddressHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FeMajAddressHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  #medians per APV for all channels, all events
  MedianAPV0HistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(256),
    Min = cms.double(0),
    Max = cms.double(1024)
    ),
  MedianAPV1HistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(256),
    Min = cms.double(0),
    Max = cms.double(1024)
    ),        
  #Sub sets of DAQ problems
  DataMissingHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadIDsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadDAQPacketHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  InvalidBuffersHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadDAQCRCsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadFEDCRCsHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  #TkHistoMap
  TkHistoMapHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  #Detailed FED level expert histograms
  FEOverflowsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  FEMissingDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadMajorityAddressesDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  BadAPVStatusBitsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  APVErrorBitsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  APVAddressErrorBitsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  UnlockedBitsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  OOSBitsDetailedHistogramConfig = cms.PSet( Enabled = cms.bool(True) ),
  #Error counting histograms
  nFEDErrorsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nFEDDAQProblemsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nFEDsWithFEProblemsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nFEDCorruptBuffersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nBadChannelStatusBitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nBadActiveChannelStatusBitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nFEDsWithFEOverflowsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nFEDsWithMissingFEsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(441),
    Min = cms.double(0),
    Max = cms.double(441)
  ),
  nUnconnectedChannelsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nAPVStatusBitHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nAPVErrorHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nAPVAddressErrorHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nUnlockedHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nOutOfSyncHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nTotalBadChannelsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  nTotalBadActiveChannelsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(353),
    Min = cms.double(0),
    Max = cms.double(42240)
  ),
  TimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(False),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nTotalBadChannelsvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nTotalBadActiveChannelsvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nFEDErrorsvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nFEDCorruptBuffersvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nFEDsWithFEProblemsvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nAPVStatusBitvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nAPVErrorvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nAPVAddressErrorvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nUnlockedvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  ),
  nOutOfSyncvsTimeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(600),
    Min = cms.double(0),
    Max = cms.double(3600)
  )
)
