import FWCore.ParameterSet.Config as cms

topElectronHLTOffDQMSource = cms.EDFilter("TopElectronHLTOfflineSource",
	DQMDirName=cms.string("HLT/TopEgOffline"),

	hltTag = cms.string("HLT"),
	superTriggerNames = cms.vstring(["HLT_L1Jet6U", "HLT_Jet15U", "HLT_QuadJet15U"]),
	electronTriggerNames = cms.vstring(["HLT_EgammaSuperClusterOnly_L1R","HLT_Ele10_LW_L1R"]),
	triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
	triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
	electronCollection = cms.InputTag("gsfElectrons"),
	primaryVertexCollection = cms.InputTag("offlinePrimaryVertices"),
	triggerJetFilterLabel =  cms.InputTag("hltL1sQuadJet15U","","HLT"),
	triggerElectronFilterLabel =  cms.InputTag("hltL1sL1SingleEG1", "", "HLT"),
	electronIdNames = cms.vstring(["eidRobustLoose"]),

	electronMinEt = cms.double(0.),
	electronMaxEta = cms.double(2.5),
	excludeCloseJets = cms.bool(True),
	requireTriggerMatch = cms.bool(False),

	addExtraId = cms.bool(True),

	# extra ID cuts - take optimised cut values (90% signal efficiency, 5% BG efficiency, TMVA method 'TMVA MC cuts')
	extraIdCutsSigmaEta = cms.double(5.3602221377786943e-03),
	extraIdCutsSigmaPhi = cms.double(6.4621652755048238e-04),
	extraIdCutsDzPV = cms.double(1.9588114237784421e-02)         
)


