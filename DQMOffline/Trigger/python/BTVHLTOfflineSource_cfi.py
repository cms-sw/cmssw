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
    turnon_threshold_loose  = cms.double(0.2),
    turnon_threshold_medium = cms.double(0.5),
    turnon_threshold_tight  = cms.double(0.8),

    pathPairs = cms.VPSet(

        cms.PSet(
            pathname = cms.string("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5_v"),
            pathtype = cms.string("PF"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5_v"),
            pathtype = cms.string("Calo"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v"),
            pathtype = cms.string("PF"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v"),
            pathtype = cms.string("Calo"),
        ),
    ),
)

#
#  Relative Online-Offline Track Monitoring
#
from DQM.TrackingMonitorSource.trackToTrackValidator_cfi import trackToTrackValidator

referenceTracksForHLTBTag = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("quality('highPurity')")
)

bTagHLTTrackMonitoring = trackToTrackValidator.clone()
bTagHLTTrackMonitoring.monitoredTrack           = cms.InputTag("hltMergedTracksForBTag")
bTagHLTTrackMonitoring.referenceTrack           = cms.InputTag("referenceTracksForHLTBTag")
bTagHLTTrackMonitoring.monitoredBeamSpot        = cms.InputTag("hltOnlineBeamSpot")
bTagHLTTrackMonitoring.referenceBeamSpot        = cms.InputTag("offlineBeamSpot")
bTagHLTTrackMonitoring.topDirName               = cms.string("HLT/BTV/HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5_v")
bTagHLTTrackMonitoring.referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices")
bTagHLTTrackMonitoring.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")
bTagHLTTrackMonitoring.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5_v*')


bTagHLTTrackMonitoringSequence = cms.Sequence(
    cms.ignore(referenceTracksForHLTBTag)
    + bTagHLTTrackMonitoring
)




