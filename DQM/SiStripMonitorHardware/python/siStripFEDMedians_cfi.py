import FWCore.ParameterSet.Config as cms

siStripFEDMedians = cms.EDAnalyzer("SiStripFEDMediansPlugin",
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),
  #Folder in DQM Store to write global histograms to
  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedMedians'),
  #Whether to dump buffer info and raw data if any error is found: 
  #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
  PrintDebugMessages = cms.untracked.uint32(1),
  #PrintDebugMessages = cms.untracked.bool(False),
  #Whether to write the DQM store to a file at the end of the run and the file name
  WriteDQMStore = cms.untracked.bool(True),
  DQMStoreFileName = cms.untracked.string('DQMStore.root'),

)
