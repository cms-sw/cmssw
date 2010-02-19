import FWCore.ParameterSet.Config as cms

topElectronHLTOffDQMClient = cms.EDFilter("TopElectronHLTOfflineClient",
	DQMDirName=cms.string("HLT/TopEgOffline"),
        runClientEndLumiBlock=cms.bool(False),
        runClientEndRun=cms.bool(True),
        runClientEndJob=cms.bool(False),

	hltTag = cms.string("HLT"),
	superTriggerNames = cms.vstring(["HLT_L1Jet6U", "HLT_Jet15U", "HLT_QuadJet15U"]),
	electronTriggerNames = cms.vstring(["HLT_EgammaSuperClusterOnly_L1R","HLT_Ele10_LW_L1R"]),
	electronIdNames = cms.vstring(["eidRobustLoose"]),
	addExtraId = cms.bool(True)
)


