import FWCore.ParameterSet.Config as cms

GLBMuonAnalyzerWithGEMs = cms.EDAnalyzer("DQMAnalyzerSTEP1",
	StandAloneTrackCollectionLabel = cms.untracked.InputTag('globalMuons','','RECO'),
	MuonSeedCollectionLabel = cms.untracked.string('standAloneMuonSeeds'),
	debug = cms.untracked.bool(False),
	folderPath = cms.untracked.string('GEMBasicPlots/SingleMu200GeV'),
	NoGEMCase = cms.untracked.bool(False),
	isGlobalMuon = cms.untracked.bool(True),
	EffSaveRootFile = cms.untracked.bool(True),
	EffRootFileName = cms.untracked.string('GLBMuonAnalyzerWithGEMs_1step_200GeV.root')
)
