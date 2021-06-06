import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPerhiphHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ["HLT_HISinglePhoton*_Eta*_Cent50_100_*",
                "HLT_HISinglePhoton*_Eta*_Cent30_100_*",
                "HLT_HIFullTrack*_L1Centrality30100_*",
                "HLT_HIPuAK4CaloJet*_Eta5p1_Cent50_100_v*",
                "HLT_HIPuAK4CaloJet*_Eta5p1_Cent30_100_v*",
                "HLT_HIDmesonHITrackingGlobal_Dpt*_Cent50_100_v*",
                "HLT_HIDmesonHITrackingGlobal_Dpt*_Cent30_100_v*",
                "HLT_HIL1Centralityext30100MinimumumBiasHF*",
                "HLT_HIL1Centralityext50100MinimumumBiasHF*",
                "HLT_HIQ2*005_Centrality3050_v*",
                "HLT_HIQ2*005_Centrality5070_v*",
                "HLT_HICastor*",
                "HLT_HIL1Castor*",
                "HLT_HIUPC*"],
    throw = False,
    andOr = True
)

peripheralHLTFilterSequence = cms.Sequence( hltPerhiphHI )
