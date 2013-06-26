import HLTrigger.HLTfilters.hltHighLevel_cfi

SiPixelLorentzAngleHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_IsoMu11', 
        'HLT_DoubleMu3', 
        'HLT_DoubleMu3_JPsi', 
        'HLT_DoubleMu3_Upsilon', 
        'HLT_DoubleMu7_Z', 
        'HLT_DoubleMu3_SameSign'],
    throw = False #dont throw except on unknown path name 
)


