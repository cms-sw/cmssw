import FWCore.ParameterSet.Config as cms

HardwareMonitor = cms.EDAnalyzer("SiStripFEDMonitorPlugin",
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),

  #Folder in DQM Store to write global histograms to
  FolderName = cms.untracked.string('SiStrip/ReadoutView/FedMonitoringSummary'),
  
  #Do not dump buffer info and raw data if any error is found
  PrintDebugMessages = cms.untracked.bool(False),
  #Do not write the DQM store to a file (DQMStore.root) at the end of the run
  WriteDQMStore = cms.untracked.bool(False),
  
  #Do book expert histograms at global level
  DisableGlobalExpertHistograms = cms.untracked.bool(False),
  #Disable the FED level histograms
  DisableFEDHistograms = cms.untracked.bool(True),
  #Book the error count histograms used for historic DQM
  DisableErrorCountHistograms = cms.untracked.bool(False),
  #Override previous two option and book and fill all histograms (so that files can be merged)
  FillAllHistograms = cms.untracked.bool(False),
  
  #Do nothing (exist for compatibility with old configs
  #rootFile = cms.untracked.string(''),
  #buildAllHistograms = cms.untracked.bool(False)
)
