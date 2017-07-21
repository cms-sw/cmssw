import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

HiJetClient = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HI/HLT_AK4Calo*", "HLT/HI/HLT_AK4PF*", "HLT/HI/HLT_HISinglePhoton*", "HLT/HI/HLT_FullTrack*", "HLT/HI/HLT_PAFullTracks*", "HLT/HI/HLT_PAAK4*", "HLT/HI/HLT_PASinglePhoton*", "HLT/HI/HLT_L1MinimumBiasHF*", "HLT/HI/HLT_ZeroBias*", "HLT/HI/HLT_PADoubleEG2*", "HLT/HI/HLT_PASingleEG5*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        #"effVsRecoPtAve 'Trigger efficiency vs reco ptAve; average p_{T}^{reco}' recoPFJetsTopology_ptAve_nominator recoPFJetsTopology_ptAve_denominator"
    ),
)
