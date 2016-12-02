import FWCore.ParameterSet.Config as cms

BTVHLTOfflineSource = cms.EDAnalyzer(
    "BTVHLTOfflineSource",
    #
    dirname = cms.untracked.string("HLT/BTV"),
    processname = cms.string("HLT"),  
    verbose = cms.untracked.bool(False),
    #
    triggerSummaryLabel    = cms.InputTag("hltTriggerSummaryAOD",                 "",    "HLT"),
    triggerResultsLabel    = cms.InputTag("TriggerResults",                       "",    "HLT"),
    triggerPathPF          = cms.string('HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq460_v5'),
    triggerFilterPFbfCSV   = cms.InputTag('hltBTagCaloCSVp022Single',             '',    'HLT'),
    #triggerFilterPFbfCSV   = cms.InputTag('hltSelector6PFJets',                   '',    'HLT'),
    triggerFilterPFafCSV   = cms.InputTag('hltBTagPFCSVp016SingleWithMatching',   '',    'HLT'),

    offlineCSVLabelPF   = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    offlineCSVLabelCalo = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    hltFastPVLabel      = cms.InputTag("hltFastPrimaryVertex"),
    hltPFPVLabel        = cms.InputTag("hltVerticesPFSelector"),
    hltCaloPVLabel      = cms.InputTag("hltVerticesL3"),    
    offlinePVLabel      = cms.InputTag("offlinePrimaryVertices"),    
    
    #
    pathPairs = cms.VPSet(
        cms.PSet(
            pathname = cms.string("HLT_QuadPFJet_BTagCSV"),
            pathtype = cms.string("PF"),
        ),
        cms.PSet(
            #pathname = cms.string("HLT_PFMET120_NoiseCleaned_BTagCSV07"),
            pathname = cms.string("HLT_PFMET120_"),
	    pathtype = cms.string("Calo"),
        )    
    )
)
