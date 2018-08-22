import FWCore.ParameterSet.Config as cms

L1TkMuons = cms.EDProducer("L1TkMuonProducer",
    ###############################################
    ############################################### common stuff    
    L1BMTFInputTag  = cms.InputTag("simBmtfDigis","BMTF"),
    L1OMTFInputTag  = cms.InputTag("simOmtfDigis","OMTF"),
    L1EMTFInputTag  = cms.InputTag("simEmtfDigis","EMTF"),
    L1EMTFTrackCollectionInputTag = cms.InputTag("simEmtfDigis"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
    ###############################################
    ############################################### TP algo
    ETAMIN = cms.double(0),
    ETAMAX = cms.double(5.),        # no cut
    ZMAX = cms.double( 25. ),       # in cm
    CHI2MAX = cms.double( 100. ),
    PTMINTRA = cms.double( 2. ),    # in GeV
    DRmax = cms.double( 0.5 ),
    nStubsmin = cms.int32( 3 ),        # minimum number of stubs
#    closest = cms.bool( True ),
    correctGMTPropForTkZ = cms.bool(True),
    use5ParameterFit = cms.bool(False), #use 4-pars by defaults
    # emtfMatchAlgoVersion = cms.int32( 1 ),        # version of matching EMTF with Trackes (1 or 2)
    emtfMatchAlgoVersion = cms.string( 'DynamicWindows' ), # version of matching EMTF with Trackes (string ID) : TP, DynamicWindows
    ###############################################
    ############################################### DynamicWindows algo
    ##### parameters for the DynamicWindows algo - eventually to put in a separate file, that will override some dummy defaults
    emtfcorr_boundaries     = cms.FileInPath('L1Trigger/L1TMuon/data/emtf_luts/matching_windows_boundaries.root'),
    emtfcorr_theta_windows  = cms.FileInPath('L1Trigger/L1TMuon/data/emtf_luts/matching_windows_theta_q99.root'),
    emtfcorr_phi_windows    = cms.FileInPath('L1Trigger/L1TMuon/data/emtf_luts/matching_windows_phi_q99.root'),
    ## block to control the evolution of the matching window vs pt
    ## if do_relax_factors = False ==> global scale of upper and lower boundaries by "final_window_factor"
    ## if do_relax_factors = True  ==> progressive linear scale, the factor is
    ##      - initial_window_factor for pt <= pt_start_relax
    ##      - final_window_factor for pt >= pt_end_relax
    ##      - and a linear interpolation in the middle
    ## facror = 0 --> no changes to the window size
    initial_window_factor   = cms.double(0.0),
    final_window_factor     = cms.double(0.0),
    pt_start_relax          = cms.double(2.0),
    pt_end_relax            = cms.double(6.0),
    do_relax_factors        = cms.bool(True),
    ##
    n_trk_par      = cms.int32(4), # 4 or 5
    min_trk_p      = cms.double(3.5),
    max_trk_aeta   = cms.double(2.5),
    max_trk_chi2   = cms.double(100.0),
    min_trk_nstubs = cms.int32(4),


)