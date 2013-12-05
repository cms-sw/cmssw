import FWCore.ParameterSet.Config as cms

DQMGEMSecondStep = cms.EDAnalyzer('DQMAnalyzerSTEP2',
  	GlobalFolder = cms.untracked.string("GEMBasicPlots/"),
  	#LocalFolder = cms.untracked.vstring("5GeV","10GeV","50GeV","100GeV","200GeV","500GeV","1000GeV"),
  	LocalFolder = cms.untracked.vstring("200GeV"),
  	SaveFile  = cms.untracked.bool(True),
  	NameFile  = cms.untracked.string("GEMPlots.root")
)
