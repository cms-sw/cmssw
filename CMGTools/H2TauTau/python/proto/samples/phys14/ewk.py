import CMGTools.RootTools.fwlite.Config as cfg

DYJets = cfg.MCComponent(
    name = 'DYJets',
    files = [],
    xSection = 2008.4*3.,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

DY1Jets = cfg.MCComponent(
    name = 'DY1Jets',
    files = [],
    xSection = 561.0,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

DY2Jets = cfg.MCComponent(
    name = 'DY2Jets',
    files = [],
    xSection = 181.,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

DY3Jets = cfg.MCComponent(
    name = 'DY3Jets',
    files = [],
    xSection = 51.1,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

DY4Jets = cfg.MCComponent(
    name = 'DY4Jets',
    files = [],
    xSection = 23.04,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

WJets = cfg.MCComponent(
    name = 'WJets',
    files = [],
    xSection = 36257.2 ,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W1234Jets = cfg.MCComponent(
    name = 'W1234Jets',
    files = [],
    xSection = 9401.8,
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W1Jets = cfg.MCComponent(
    name = 'W1Jets',
    files = [],
    xSection = 6440.4, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W2Jets = cfg.MCComponent(
    name = 'W2Jets',
    files = [],
    xSection = 2087.2, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W3Jets = cfg.MCComponent(
    name = 'W3Jets',
    files = [],
    xSection = 619.0, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W1Jets_ext = cfg.MCComponent(
    name = 'W1Jets_ext',
    files = [],
    xSection = 6440.4, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W2Jets_ext = cfg.MCComponent(
    name = 'W2Jets_ext',
    files = [],
    xSection = 2087.2, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W3Jets_ext = cfg.MCComponent(
    name = 'W3Jets_ext',
    files = [],
    xSection = 619.0, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

W4Jets = cfg.MCComponent(
    name = 'W4Jets',
    files = [],
    xSection = 255.2, #PG inclusive scaled according to LO XS
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

TTJets = cfg.MCComponent(
    name = 'TTJets',
    files = [],
    xSection = 228.4, # correction factor from Valentina removed as it depends on the jet binning
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

TTJetsFullLept = cfg.MCComponent(
    name = 'TTJetsFullLept',
    files = [],
    xSection = 249.5*0.96*(1-0.676)*(1-0.676), # TOP-12-007 + Valentina's SF
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

TTJetsSemiLept = cfg.MCComponent(
    name = 'TTJetsSemiLept',
    files = [],
    xSection = 249.5*0.96*(1-0.676)*0.676*2, # TOP-12-007 + Valentina's SF
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

TTJetsHadronic = cfg.MCComponent(
    name = 'TTJetsHadronic',
    files = [],
    xSection = 249.5*0.96*0.676*0.676, # TOP-12-007 + Valentina's SF
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

T_tW = cfg.MCComponent(
    name = 'T_tW',
    files = [],
    xSection = 11.1, # from the sync twiki
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )

Tbar_tW = cfg.MCComponent(
    name = 'Tbar_tW',
    files = [],
    xSection = 11.1, # from the sync twiki
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )



mc_dy = [
    DYJets,
    DY1Jets,
    DY2Jets,
    DY3Jets,
    DY4Jets,
    ]

mc_w = [
    WJets,
    W1234Jets,
    W1Jets,
    W2Jets,
    W3Jets,
    W4Jets,
    ]

mc_w_ext = [
    W1Jets_ext,
    W2Jets_ext,
    W3Jets_ext,
    ]

t_mc_ewk = [
    TTJets,
    T_tW,
    Tbar_tW,
    TTJetsFullLept,
    TTJetsSemiLept,
    TTJetsHadronic,
    ]


mc_ewk = []
mc_ewk += mc_dy
mc_ewk += mc_w
mc_ewk += t_mc_ewk


# for backward compatibility:
ztt_mc_ewk = mc_dy
ztt_inc_mc_ewk = [DYJets]
w_mc_ewk = mc_w


#stitching:

# from COLIN, measured on inclusive DYJets sample, before any selection.
dy_fractions = [ 0.72328,
                 0.188645,
                 0.0613196,
                 0.0188489,
                 0.00790643
                 ]

for dy in mc_dy:
    dy.fractions = dy_fractions


# from Jose
w_fractions = [ 0.74392452,
                0.175999,
                0.0562617,
                0.0168926,
                0.00692218
                ]

for w in mc_w + mc_w_ext:
    w.fractions = w_fractions

