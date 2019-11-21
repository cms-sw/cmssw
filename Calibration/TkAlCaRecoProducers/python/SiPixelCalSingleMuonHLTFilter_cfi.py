import HLTrigger.HLTfilters.hltHighLevel_cfi

SiPixelCalSingleMuonHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_*'],
    throw = False #dont throw except on unknown path name 
)