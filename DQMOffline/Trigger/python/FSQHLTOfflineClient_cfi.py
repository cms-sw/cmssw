import FWCore.ParameterSet.Config as cms

fsqClient = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/FSQ/HLT_DiPFJetAve*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effVsRecoPtAve 'Trigger efficiency vs reco ptAve; average p_{T}^{reco}' recoPFJetsTopology_ptAve_nominator recoPFJetsTopology_ptAve_denominator"
    ),
)

