import FWCore.ParameterSet.Config as cms

BTVHLTOfflineSource = cms.EDAnalyzer(
    "BTVHLTOfflineSource",
    #
    dirname = cms.untracked.string("HLT/BTV"),
    processname = cms.string("HLT"),  
    verbose = cms.untracked.bool(False),
    #
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    offlineCSVLabelPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    offlineCSVLabelCalo = cms.InputTag("combinedInclusiveSecondaryVertexV2BJetTags"),
    hltFastPVLabel = cms.InputTag("hltFastPrimaryVertex"),
    hltPFPVLabel = cms.InputTag("hltVerticesPFSelector"),
    hltCaloPVLabel = cms.InputTag("hltVerticesL3"),    
    offlinePVLabel = cms.InputTag("offlinePrimaryVertices"),    
    
    #
    pathPairs = cms.VPSet(
        cms.PSet(
            pathname = cms.string("HLT_QuadPFJet_SingleBTagCSV_VBF"),
            pathtype = cms.string("PF"),
        ),
        cms.PSet(
            #pathname = cms.string("HLT_PFMET120_NoiseCleaned_BTagCSV07"),
            pathname = cms.string("HLT_PFMET120_"),
	    pathtype = cms.string("Calo"),
        )    
    )
)
