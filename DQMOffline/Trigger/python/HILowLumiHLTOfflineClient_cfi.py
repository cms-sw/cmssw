import FWCore.ParameterSet.Config as cms

HiJetClient = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HI/HLT_AK4Calo*", "HLT/HI/HLT_AK4PF*", "HLT/HI/HLT_HISinglePhoton*", "HLT/HI/HLT_FullTrack*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        #"effVsRecoPtAve 'Trigger efficiency vs reco ptAve; average p_{T}^{reco}' recoPFJetsTopology_ptAve_nominator recoPFJetsTopology_ptAve_denominator"
    ),
)
