import FWCore.ParameterSet.Config as cms

fsqClient = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/FSQ/HLT_DiPFJetAve*", "HLT/FSQ/HLT_PixelTracks_Multiplicity*",\
                                           "HLT/FSQ/HLT_ZeroBias_SinglePixelTrack*"  ),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effVsRecoPtAve 'Trigger efficiency vs reco ptAve; average p_{T}^{reco}' recoPFJetsTopology_ptAve_nominator recoPFJetsTopology_ptAve_denominator",
        "effVsRecoTracksCnt 'Trigger efficiency vs reco tracks count' recoTracks_count_nominator recoTracks_count_denominator",
        "effVsAtLeastOneTrack 'Trigger efficiency for events with at least one offline track' zb_Eff_nominator zb_Eff_denominator"

    ),
)

