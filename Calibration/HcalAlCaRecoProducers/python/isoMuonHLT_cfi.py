import HLTrigger.HLTfilters.hltHighLevel_cfi

isoMuonHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
  
    HLTPaths = ['HLT_L1MuOpen','HLT_IsoMu11'],
    throw = False #dont throw except on unknown path name

)


