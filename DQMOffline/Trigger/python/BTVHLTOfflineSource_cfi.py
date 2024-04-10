import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

BTVHLTOfflineSource = DQMEDAnalyzer("BTVHLTOfflineSource",

    dirname                 = cms.untracked.string("HLT/BTV"),
    processname             = cms.string("HLT"),
    verbose                 = cms.untracked.bool(False),

    triggerSummaryLabel     = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    triggerResultsLabel     = cms.InputTag("TriggerResults", "", "HLT"),
    onlineDiscrLabelPF      = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
    onlineDiscrLabelCalo    = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
    offlineDiscrLabelb      = cms.InputTag("pfDeepCSVJetTags", "probb"),
    offlineDiscrLabelbb     = cms.InputTag("pfDeepCSVJetTags", "probbb"),
    hltFastPVLabel          = cms.InputTag("hltFastPrimaryVertex"),
    hltPFPVLabel            = cms.InputTag("hltVerticesPFSelector"),
    hltCaloPVLabel          = cms.InputTag("hltVerticesL3"),
    offlinePVLabel          = cms.InputTag("offlinePrimaryVertices"),
    offlineIPLabel          = cms.InputTag("pfImpactParameterTagInfos"),
    turnon_threshold_loose  = cms.double(0.2),
    turnon_threshold_medium = cms.double(0.5),
    turnon_threshold_tight  = cms.double(0.8),
    minDecayLength          = cms.double(-9999.0),
    maxDecayLength          = cms.double(5.0),
    minJetDistance          = cms.double(0.0),
    maxJetDistance          = cms.double(0.07),
    dRTrackMatch            = cms.double(0.01),

    pathPairs = cms.VPSet(

        cms.PSet(
            pathname = cms.string("HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71_v"),
            pathtype = cms.string("PF")
        ),
        cms.PSet(
            pathname = cms.string("HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71_v"),
            pathtype = cms.string("Calo")
        ),
   ),
)

#
#  Relative Online-Offline Track Monitoring
#
from DQM.TrackingMonitorSource.TrackToTrackComparisonHists_cfi import TrackToTrackComparisonHists

referenceTracksForHLTBTag = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("quality('highPurity')")
)

bTagHLTTrackMonitoring_muPF1 = TrackToTrackComparisonHists.clone(
    dzWRTPvCut               = 0.1,
    monitoredTrack           = "hltMergedTracks",
    referenceTrack           = "referenceTracksForHLTBTag",
    monitoredBeamSpot        = "hltOnlineBeamSpot",
    referenceBeamSpot        = "offlineBeamSpot",
    topDirName               = "HLT/BTV/HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71PF",
    referencePrimaryVertices = "offlinePrimaryVertices",
    monitoredPrimaryVertices = "hltVerticesPFSelector",
    genericTriggerEventPSet = dict(hltPaths = ["HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71*"])
)

bTagHLTTrackMonitoring_muPF2 = bTagHLTTrackMonitoring_muPF1.clone( 
    topDirName = "HLT/BTV/HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71PF",
    genericTriggerEventPSet = dict(hltPaths = ["HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71*"])
)

bTagHLTTrackMonitoringSequence = cms.Sequence(
    cms.ignore(referenceTracksForHLTBTag)
    + bTagHLTTrackMonitoring_muPF1
    + bTagHLTTrackMonitoring_muPF2
)

