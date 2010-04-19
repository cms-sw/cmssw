import FWCore.ParameterSet.Config as cms

siStripFEDCheck = cms.EDAnalyzer("SiStripFEDCheckPlugin",
  #Directory to book histograms in
  DirName = cms.string('SiStrip/FEDIntegrity/'),
  #Raw data collection
  RawDataTag = cms.InputTag('source'),
  #Number of events to cache info before updating histograms
  #(set to zero to disable cache)
  #HistogramUpdateFrequency = cms.uint32(0),
  HistogramUpdateFrequency = cms.uint32(1000),
  #Print info about errors buffer dumps to LogInfo(SiStripFEDCheck)
  PrintDebugMessages = cms.bool(False),
  #Write the DQM store to a file (DQMStore.root) at the end of the run
  WriteDQMStore = cms.bool(False),
  #Use to disable all payload (non-fatal) checks
  DoPayloadChecks = cms.bool(True),
  #Use to disable check on channel lengths
  CheckChannelLengths = cms.bool(True),
  #Use to disable check on channel packet codes
  CheckChannelPacketCodes = cms.bool(True),
  #Use to disable check on FE unit lengths in full debug header
  CheckFELengths = cms.bool(True),
  #Use to disable check on channel status bits
  CheckChannelStatus = cms.bool(True),
)
