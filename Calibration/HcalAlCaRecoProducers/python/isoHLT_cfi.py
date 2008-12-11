import HLTrigger.HLTfilters.hltHighLevel_cfi

isoHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['AlCa_IsoTrack'],
    throw = False #dont throw except on unknown path name 

)


