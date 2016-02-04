import HLTrigger.HLTfilters.hltHighLevel_cfi

l1RPCHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_L1Mu'],
    throw = False #dont throw except on unknown path names
)


