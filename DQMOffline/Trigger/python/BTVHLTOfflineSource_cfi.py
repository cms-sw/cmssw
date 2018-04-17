import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
BTVHLTOfflineSource = DQMEDAnalyzer(
    "BTVHLTOfflineSource",
    #
    dirname                 = cms.untracked.string("HLT/BTV"),
    processname             = cms.string("HLT"),
    verbose                 = cms.untracked.bool(False),
    #
    triggerSummaryLabel     = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel     = cms.InputTag("TriggerResults","","HLT"),
    onlineDiscrLabelPF      = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
    onlineDiscrLabelCalo    = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
    offlineDiscrLabelb      = cms.InputTag("pfDeepCSVJetTags", "probb"),
    offlineDiscrLabelbb     = cms.InputTag("pfDeepCSVJetTags", "probbb"),
    hltFastPVLabel          = cms.InputTag("hltFastPrimaryVertex"),
    hltPFPVLabel            = cms.InputTag("hltVerticesPFSelector"),
    hltCaloPVLabel          = cms.InputTag("hltVerticesL3"),
    offlinePVLabel          = cms.InputTag("offlinePrimaryVertices"),
    turnon_threshold_loose  = cms.double(0.2),
    turnon_threshold_medium = cms.double(0.5),
    turnon_threshold_tight  = cms.double(0.8),
    #
    pathPairs = cms.VPSet(
        cms.PSet(
            pathname = cms.string("HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_"),
            pathtype = cms.string("PF"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_"),
            pathtype = cms.string("Calo"),
        )
    )
)

btvHLTDQMSourceExtra = cms.Sequence(
)
