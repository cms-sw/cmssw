import HLTrigger.HLTfilters.hltHighLevel_cfi

isoMuonHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
  
    HLTPaths = ['HLT_L1MuOpen','HLT_IsoMu11','HLT_L2Mu9','HLT_Mu5','HLT_Mu9'],
    throw = False #dont throw except on unknown path name

)


