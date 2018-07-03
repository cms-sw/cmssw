import HLTrigger.HLTfilters.hltHighLevel_cfi

SiPixelLorentzAngleHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    # FIXME: check with package maintainers which set of paths to put in the trigger bits payload
    #        the ones below or the ones in 'Calibration/TkAlCaRecoProducers/python/ALCARECOSiPixelLorentzAngle_cff.py'
    # HLTPaths = ['HLT1MuonIso',
    #     'HLT2MuonNonIso',
    #     'HLT2MuonJPsi',
    #     'HLT2MuonUpsilon',
    #     'HLT2MuonZ',
    #     'HLT2MuonSameSign'],
    eventSetupPathsKey = 'SiPixelLorentzAngle',
    throw = False #dont throw except on unknown path names
)


