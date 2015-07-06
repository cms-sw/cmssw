from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass,  binning_svfitMass_finer


class PlotInfo(object):
    def __init__(self, var, nbins, xmin, xmax):
        self.var = var
        self.nbins = nbins
        self.xmin = xmin
        self.xmax = xmax
        

plots_All = {

    'l2_pt': PlotInfo('l2_pt', 20, 0, 200),
    'l2_eta': PlotInfo('l2_eta', 50, -3, 3),
    'l2_relIso05': PlotInfo('l2_relIso05', 50, 0, 0.5),
    'l2_charge': PlotInfo('l2_charge', 3, -1.5, 1.5),
    'l1_pt': PlotInfo('l1_pt', 20, 0, 200),
    'l1_eta': PlotInfo('l1_eta', 50, -3, 3),
    'l1_charge': PlotInfo('l1_charge', 3, -1.5, 1.5),
    'l1_threeHitIso': PlotInfo('l1_threeHitIso', 25, 0., 10.),
    'l1_decayMode':PlotInfo('l1_decayMode', 15, -0.5, 14.5),
    'l1_mass':PlotInfo('l1_mass', 30, 0., 2.),
    'mt': PlotInfo('mt', 10, 0, 150),
    'visMass': PlotInfo('visMass', 50, 0, 200),
    'svfitMass': PlotInfo('svfitMass', 50, 0, 300),
    'svfitMassFewerBins': PlotInfo('svfitMass', 30, 0, 300),
    'svfitMassFewBins': PlotInfo('svfitMass', 10, 0, 300),
    'nJets': PlotInfo('nJets', 7, -0.5, 6.5),
    # 'nJets20': PlotInfo('nJets20', 10, -0.5, 9.5),
    # 'nVert': PlotInfo('nVert', 50, 0, 50),
    'met': PlotInfo('met', 50, 0, 150),
    # 'pfmet': PlotInfo('met', 50, 0, 120),
    'diTau_pt': PlotInfo('diTau_pt', 50, 0, 150),
    # 'bjet1_pt': PlotInfo('bjet1_pt', 50, 0, 200),
    'nVert': PlotInfo('nVert', 51, -0.5, 50.5),
    'pthiggs': PlotInfo('pthiggs', 50, 0., 150.),
    'deltaEtaL1L2': PlotInfo('deltaEtaL1L2', 20, 0., 5.),
    'deltaRL1L2': PlotInfo('deltaRL1L2', 20, 0., 5.),
    #'abs_l1_eta_j1_eta': PlotInfo('abs_l1_eta_j1_eta', 20, 0., 5.),
    #'abs_l2_eta_j1_eta': PlotInfo('abs_l2_eta_j1_eta', 20, 0., 5.),
    'deltaPhiL1L2' :
        PlotInfo('deltaPhiL1L2', 25, -3.141593, 3.141593),
    'deltaPhiL1LMET' :
        PlotInfo('deltaPhiL1MET', 25, -3.141593, 3.141593),
    'deltaPhiL2LMET' :
        PlotInfo('deltaPhiL2MET', 25, -3.141593, 3.141593),
    'deltaRL1L2' :
        PlotInfo('deltaRL1L2', 25, 0., 5.),
    # 'deltaPhi_l1_j1' :
        # PlotInfo('deltaPhi_l1_j1', 25, 0., 3.141593),
    # 'deltaPhi_j1_met' :
        # PlotInfo('deltaPhi_j1_met', 25, 0., 3.141593),
    # 'deltaPhi_l2_j1' :
        # PlotInfo('deltaPhi_l2_j1', 25, 0., 3.141593),
    'jet1_pt': PlotInfo('jet1_pt', 30, 0, 300),
    'bjet1_pt': PlotInfo('bjet1_pt', 30, 0, 300),
    
    }

plots_All_sorted_indices = [
    'l2_pt',
    'l2_eta',
    'l2_relIso05',
    'l2_charge',
    'l1_pt',
    'l1_eta',
    'l1_charge',
    'l1_threeHitIso',
    'l1_decayMode',
    'l1_mass',
    'mt',
    'met',
    'diTau_pt',
    'pthiggs',
    'visMass',
    'svfitMass',
    'svfitMassFewerBins',
    'svfitMassFewBins',
    'nJets',
    'jet1_pt',
    'bjet1_pt',
    'nVert',
    'deltaEtaL1L2',
    'deltaRL1L2',
    #'abs_l1_eta_j1_eta',
    #'abs_l2_eta_j1_eta',
    'deltaPhiL1L2' ,
    'deltaPhiL1LMET',
    'deltaPhiL2LMET' ,
    'deltaRL1L2' ,
    # 'deltaPhi_l1_j1',
    # 'deltaPhi_l2_j1',
    ]

plots_J0 = {
    'l2_pt': PlotInfo('l2_pt', 100, 0, 200),
    'l2_eta': PlotInfo('l2_eta', 50, -3, 3),
    'l2_relIso05': PlotInfo('l2_relIso05', 50, 0, 0.1),
    'l1_pt': PlotInfo('l1_pt', 100, 0, 200),
    'l1_eta': PlotInfo('l1_eta', 50, -3, 3),
    # 'mt': PlotInfo('mt', 50, 0, 150),
    'visMass': PlotInfo('visMass', binning_svfitMass_finer, None, None),
    'svfitMass': PlotInfo('svfitMass', binning_svfitMass_finer, None, None),
    }

plots_J0_high = {
    'l2_pt': PlotInfo('l2_pt', 50, 0, 200),
    'l2_eta': PlotInfo('l2_eta', 20, -3, 3),
    'l2_relIso05': PlotInfo('l2_relIso05', 50, 0, 0.1),
    'l1_pt': PlotInfo('l1_pt', 50, 0, 200),
    'l1_eta': PlotInfo('l1_eta', 20, -3, 3),
    # 'mt': PlotInfo('mt', 50, 0, 150),
    'visMass': PlotInfo('visMass', binning_svfitMass_finer, None, None),
    'svfitMass': PlotInfo('svfitMass', binning_svfitMass_finer, None, None),
    }

plots_J1 = {
    'l2_pt': PlotInfo('l2_pt', 40, 0, 200),
    'l2_eta': PlotInfo('l2_eta', 20, -3, 3),
    'l2_relIso05': PlotInfo('l2_relIso05', 50, 0, 0.1),
    'l1_pt': PlotInfo('l1_pt', 40, 0, 200),
    'l1_eta': PlotInfo('l1_eta', 20, -3, 3),
    # 'mt': PlotInfo('mt', 25, 0, 150),
    'visMass': PlotInfo('visMass', binning_svfitMass_finer, None, None),
    'svfitMass': PlotInfo('svfitMass', binning_svfitMass_finer, None, None),
    'jet1_pt': PlotInfo('jet1_pt', 30, 0, 300),
    'jet1_eta': PlotInfo('jet1_eta', 25, -5, 5),
    }

plots_J1_high = {
    'l2_pt': PlotInfo('l2_pt', 40, 0, 200),
    'l2_eta': PlotInfo('l2_eta', 20, -3, 3),
    'l2_relIso05': PlotInfo('l2_relIso05', 50, 0, 0.1),
    'l1_pt': PlotInfo('l1_pt', 40, 0, 200),
    'l1_eta': PlotInfo('l1_eta', 20, -3, 3),
    # 'mt': PlotInfo('mt', 20, 0, 150),
    'visMass': PlotInfo('visMass', binning_svfitMass_finer, None, None),
    'svfitMass': PlotInfo('svfitMass', binning_svfitMass_finer, None, None),
    'jet1_pt': PlotInfo('jet1_pt', 30, 0, 300),
    'jet1_eta': PlotInfo('jet1_eta', 25, -5, 5),
    }

plots_J2 = {
    'jet1_pt': PlotInfo('jet1_pt', 30, 0, 300),
    'jet1_eta': PlotInfo('jet1_eta', 10, -5, 5),
    'jet2_pt': PlotInfo('jet1_pt', 30, 0, 300),
    'jet2_eta': PlotInfo('jet1_eta', 10, -5, 5),
    'svfitMass': PlotInfo('svfitMass', 30, 0, 300)    
    }

plots_TauEle_basic = {
    'l1_pt'      : PlotInfo ('l1_pt',       25,  0,    100), # tau
    'l1_eta'     : PlotInfo ('l1_eta',      40,  -2.5, 2.5),
    'l2_pt'      : PlotInfo ('l2_pt',       25,  0,    100), # electron
    'l2_eta'     : PlotInfo ('l2_eta',      40,  -2.5, 2.5),
    'mt'         : PlotInfo ('mt',          40,  0,    200),
    'met'        : PlotInfo ('met',         30,  0,    150),
    'svfitMass'  : PlotInfo ('svfitMass',   30,  0,    300),
    'nJets'      : PlotInfo ('nJets',       10,  0,     10),
    }

plots_TauEle = {
    'l1_pt'      : PlotInfo ('l1_pt',       25,  0,    100), # tau
    'l1_eta'     : PlotInfo ('l1_eta',      40,  -2.5, 2.5),
    'l1_phi'     : PlotInfo ('l1_phi',      20,  -3.5, 3.5),
    'l1_charge'  : PlotInfo ('l1_charge',   3,   -1.5, 1.5),
# tau iso missing
    'l1_EOverp'  : PlotInfo ('l1_EOverp',   100, 0,    2  ),
    'l2_pt'      : PlotInfo ('l2_pt',       25,  0,    100), # electron
    'l2_eta'     : PlotInfo ('l2_eta',      40,  -2.5, 2.5),
    'l2_phi'     : PlotInfo ('l2_phi',      20,  -3.5, 3.5),
    'l2_charge'  : PlotInfo ('l2_charge',   3,   -1.5, 1.5),
    'l2_relIso05': PlotInfo ('l2_relIso05', 50,  0,      1),
    'mt'         : PlotInfo ('mt',          40,  0,    200),
    'met'        : PlotInfo ('met',         30,  0,    150),
    'svfitMass'  : PlotInfo ('svfitMass',   30,  0,    300),
    'nJets'      : PlotInfo ('nJets',       10,  0,     10),
#    'visMass'    : PlotInfo ('visMass',     50,  0,    200), 
#    'pfmet'      : PlotInfo ('pfmet',       50,  0,    150),
#    'nVert'      : PlotInfo ('nVert',       50,  0,    50 ),
    }
