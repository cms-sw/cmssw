import HLTrigger.HLTfilters.hltHighLevel_cfi

SiPixelLorentzAngleHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT1MuonIso', 
        'HLT2MuonNonIso', 
        'HLT2MuonJPsi', 
        'HLT2MuonUpsilon', 
        'HLT2MuonZ', 
        'HLT2MuonSameSign'],
    throw = False #dont throw except on unknown path names
)


