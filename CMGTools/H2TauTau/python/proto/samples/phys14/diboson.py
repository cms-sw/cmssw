import CMGTools.RootTools.fwlite.Config as cfg

# exclusive madgraph samples
# -- -- -- -- -- -- -- -- 

WWJetsTo2L2Nu = cfg.MCComponent(
    name = 'WWJetsTo2L2Nu',
    files = [],
    xSection = 5.824, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

WZJetsTo2L2Q = cfg.MCComponent(
    name = 'WZJetsTo2L2Q',
    files = [],
    xSection = 2.207, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

WZJetsTo3LNu = cfg.MCComponent(
    name = 'WZJetsTo3LNu',
    files = [],
    xSection = 1.058, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

ZZJetsTo2L2Nu = cfg.MCComponent(
    name = 'ZZJetsTo2L2Nu',
    files = [],
    xSection = 0.716, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

ZZJetsTo2L2Q = cfg.MCComponent(
    name = 'ZZJetsTo2L2Q',
    files = [],
    xSection = 2.502, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

ZZJetsTo4L = cfg.MCComponent(
    name = 'ZZJetsTo4L',
    files = [],
    xSection = 0.181, #PG from twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorkingSummer2012#MC_samples_and_cross_sections
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

mc_diboson_xcl = [
    WWJetsTo2L2Nu,
    WZJetsTo2L2Q,
    WZJetsTo3LNu,
    ZZJetsTo2L2Nu,
    ZZJetsTo2L2Q,
    ZZJetsTo4L
    ]


# inclusive pythia samples
# -- -- -- -- -- -- -- -- 

WW = cfg.MCComponent(
    name = 'WW',
    files = [],
#    xSection = 57.1097, # correction factor from Valentina
    xSection = 54.838, #PG numbers from Andrew
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )


WZ = cfg.MCComponent(
    name = 'WZ',
    files = [],
#    xSection = 32.3161,
#    xSection = 32.3161 * 0.97, #PG scale factor wrt exclusive samples XS
    xSection = 33.21, #PG number from Andrew
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )


ZZ = cfg.MCComponent(
    name = 'ZZ',
    files = [],
#    xSection = 8.25561, # correction factor from Valentina
#    xSection = 8.3 * 2.13, #PG scale factor wrt exclusive samples XS
    xSection = 17.654, #PG number from Andrew
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )


mc_diboson = [
    WWJetsTo2L2Nu,
    WZJetsTo2L2Q,
    WZJetsTo3LNu,
    ZZJetsTo2L2Nu,
    ZZJetsTo2L2Q,
    ZZJetsTo4L,
    # WW,
    # WZ,
    # ZZ
    ]


mc_diboson_incl = [
    WW,
    WZ,
    ZZ
    ]

