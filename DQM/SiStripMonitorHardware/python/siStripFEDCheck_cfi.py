import FWCore.ParameterSet.Config as cms

siStripFEDCheck = cms.EDAnalyzer("SiStripFEDCheckPlugin",
  #Directory to book histograms in
  DirName = cms.untracked.string('SiStrip/FEDIntegrity/'),
  #Raw data collection
  RawDataTag = cms.InputTag('source'),
  #Number of events to cache info before updating histograms
  #(set to zero to disable cache)
  #HistogramUpdateFrequency = cms.untracked.uint32(0),
  HistogramUpdateFrequency = cms.untracked.uint32(1000),
  #Print info about errors buffer dumps to LogInfo(SiStripFEDCheck)
  PrintDebugMessages = cms.untracked.bool(False),
  doPLOTfedsPresent       = cms.bool(True),                                 
  doPLOTfedFatalErrors    = cms.bool(True),                                 
  doPLOTfedNonFatalErrors = cms.bool(True),                                 
  doPLOTnFEDinVsLS        = cms.bool(False),                                 
  doPLOTnFEDinWdataVsLS   = cms.bool(False),                                 
  #Write the DQM store to a file (DQMStore.root) at the end of the run
  WriteDQMStore = cms.untracked.bool(False),
  #Use to disable all payload (non-fatal) checks
  DoPayloadChecks = cms.untracked.bool(True),
  #Use to disable check on channel lengths
  CheckChannelLengths = cms.untracked.bool(True),
  #Use to disable check on channel packet codes
  CheckChannelPacketCodes = cms.untracked.bool(True),
  #Use to disable check on FE unit lengths in full debug header
  CheckFELengths = cms.untracked.bool(True),
  #Use to disable check on channel status bits
  CheckChannelStatus = cms.untracked.bool(True),
  LSBin = cms.int32(5000),
  LSMin = cms.double(0.5),
  LSMax = cms.double(5000.5),  
)
