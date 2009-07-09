import FWCore.ParameterSet.Config as cms

siStripFEDMonitor = cms.EDAnalyzer("SiStripFEDMonitorPlugin",
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),
  #Folder in DQM Store to write global histograms to
  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedMonitoringSummary'),
  #Fill all detailed histograms at FED level even if they will be empty (so that files can be merged)
  FillAllDetailedHistograms = cms.untracked.bool(False),
  #do histos vs time with time=event number. Default time = orbit number (s).
  FillWithEventNumber = cms.untracked.bool(False),
  #Whether to dump buffer info and raw data if any error is found: 
  #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
  PrintDebugMessages = cms.untracked.uint32(1),
  #PrintDebugMessages = cms.untracked.bool(False),
  #Whether to write the DQM store to a file at the end of the run and the file name
  WriteDQMStore = cms.untracked.bool(False),
  DQMStoreFileName = cms.untracked.string('DQMStore.root'),

  #Historgram configuration
  #Global/summary histograms
  DataPresentHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  AnyFEDErrorsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  AnyDAQProblemsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  AnyFEProblemsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  CorruptBuffersHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadChannelStatusBitsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadActiveChannelStatusBitsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  #Sub sets of FE problems
  FEOverflowsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  FEMissingHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadMajorityAddressesHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  #Sub sets of DAQ problems
  DataMissingHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadIDsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadDAQPacketHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  InvalidBuffersHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadDAQCRCsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadFEDCRCsHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  #TkHistoMap
  TkHistoMapHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  #Detailed FED level expert histograms
  FEOverflowsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  FEMissingDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadMajorityAddressesDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  BadAPVStatusBitsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  APVErrorBitsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  APVAddressErrorBitsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  UnlockedBitsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  OOSBitsDetailedHistogramConfig = cms.untracked.PSet( Enabled = cms.untracked.bool(True) ),
  #Error counting histograms
  nFEDErrorsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nFEDDAQProblemsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nFEDsWithFEProblemsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nFEDCorruptBuffersHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nBadChannelStatusBitsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nBadActiveChannelStatusBitsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nFEDsWithFEOverflowsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nFEDsWithMissingFEsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(441),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(441)
  ),
  nUnconnectedChannelsHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nAPVStatusBitHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nAPVErrorHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nAPVAddressErrorHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nUnlockedHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nOutOfSyncHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(353),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(42240)
  ),
  nTotalBadChannelsvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nTotalBadActiveChannelsvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nFEDErrorsvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nFEDCorruptBuffersvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nFEDsWithFEProblemsvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nAPVStatusBitvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nAPVErrorvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nAPVAddressErrorvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nUnlockedvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  ),
  nOutOfSyncvsTimeHistogramConfig = cms.untracked.PSet(
    Enabled = cms.untracked.bool(True),
    NBins = cms.untracked.uint32(600),
    Min = cms.untracked.double(0),
    Max = cms.untracked.double(3600)
  )
)
