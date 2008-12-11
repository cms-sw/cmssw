import HLTrigger.HLTfilters.hltHighLevel_cfi

isoHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['CandHLTHcalIsolatedTrackNoEcalIsol'],
    andOr = cms.bool(True),
    throw = False #dont throw except on unknown path name 
)


