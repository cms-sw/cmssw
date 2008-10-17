import FWCore.ParameterSet.Config as cms

siStripFEDCheck = cms.EDAnalyzer("SiStripFEDCheckPlugin",
  FolderName = cms.untracked.string('SiStrip/FEDIntegrity/'),
  RawDataTag = cms.untracked.InputTag('source'),
  PrintDebugMessages = cms.untracked.bool(False),
  WriteDQMStore = cms.untracked.bool(False),
)
