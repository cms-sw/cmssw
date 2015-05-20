import FWCore.ParameterSet.Config as cms

#print "please remember to switch off gen jet efficiency plots in client"
fsqClient = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/FSQ/HLT_DiPFJetAve*", "HLT/FSQ/HLT_PixelTracks_Multiplicity*",\
                                           "HLT/FSQ/HLT_ZeroBias_SinglePixelTrack*", \
                                           "HLT/FSQ/HLT_PFJet*", "HLT/FSQ/HLT_DiPFJet*" ),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        #"effVsHardestGenJetPT 'Trigger efficiency vs hardest gen jet pT; p_{T}^{gen}' genJets_pt_nominator genJets_pt_denominator",
        #"effVsSofterGenJetPT 'Trigger efficiency vs hardest gen jet pT; p_{T}^{gen}' genJets_ptm_nominator genJets_ptm_denominator",
        "effVsRecoPtAve 'Trigger efficiency vs reco ptAve; average p_{T}^{reco}' recoPFJetsTopology_ptAve_nominator recoPFJetsTopology_ptAve_denominator",
        "effVsRecoTracksCnt 'Trigger efficiency vs reco tracks count' recoTracks_count_nominator recoTracks_count_denominator",
        "effVsAtLeastOneTrack 'Trigger efficiency for events with at least one offline track' zb_Eff_nominator zb_Eff_denominator",
        "effVsHardestAK4PFchs 'Trigger efficiency vs hardest reco jet pT; p_{T}^{rec}' ak4PFJetsCHS_pt_nominator ak4PFJetsCHS_pt_denominator",
        "effVsSofterAK4PFchs 'Trigger efficiency vs hardest reco jet pT; p_{T}^{rec}' ak4PFJetsCHS_ptm_nominator ak4PFJetsCHS_ptm_denominator"
    ),
)

