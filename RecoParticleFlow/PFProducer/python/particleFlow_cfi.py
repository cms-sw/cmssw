import FWCore.ParameterSet.Config as cms

particleFlowTmp = cms.EDProducer("PFProducer",

    # Verbose and debug flags
    verbose = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),

    # PF Blocks label
    blocks = cms.InputTag("particleFlowBlock"),

    # reco::muons label and Post Muon cleaning
    muons = cms.InputTag("muons1stStep"),
    postMuonCleaning = cms.bool(True), # Propagated to PFMuonAlgo

    # Vertices label
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
    useVerticesForNeutral = cms.bool(True),

    # Use HO clusters in PF hadron reconstruction
    useHO = cms.bool(True),                                 

    # EGamma-related
    PFEGammaCandidates = cms.InputTag("particleFlowEGamma"),
    GedElectronValueMap = cms.InputTag("gedGsfElectronsTmp"),
    GedPhotonValueMap = cms.InputTag("gedPhotonsTmp","valMapPFEgammaCandToPhoton"),

    useEGammaElectrons = cms.bool(True),
    egammaElectrons = cms.InputTag('mvaElectrons'),

    useEGammaFilters = cms.bool(True),
    useProtectionsForJetMET = cms.bool(True), # Propagated to PFEGammaFilters

    #----------------------------------
    # For PFEGammaFilters
    #----------------------------------
    PFEGammaFiltersParameters = cms.PSet(

        # New electron selection cuts for CMSSW_700
        electron_iso_pt = cms.double(10.0),
        electron_iso_mva_barrel  = cms.double( -0.1875),
        electron_iso_mva_endcap = cms.double( -0.1075),
        electron_iso_combIso_barrel = cms.double(10.0),
        electron_iso_combIso_endcap = cms.double(10.0),
        electron_noniso_mvaCut = cms.double(-0.1),
        electron_missinghits = cms.uint32(1),
        electron_ecalDrivenHademPreselCut = cms.double(0.15),
        electron_maxElePtForOnlyMVAPresel = cms.double(50.),
        electron_protectionsForJetMET = cms.PSet(
            maxNtracks = cms.double(3.0), #max tracks pointing at Ele cluster
            maxHcalE = cms.double(10.0),
            maxTrackPOverEele = cms.double(1.0),
            maxE = cms.double(50.0), #for dphi cut
            maxEleHcalEOverEcalE = cms.double(0.1),
            maxEcalEOverPRes = cms.double(0.2),
            maxEeleOverPoutRes = cms.double(0.5),
            maxHcalEOverP = cms.double(1.0),
            maxHcalEOverEcalE = cms.double(0.1),
            maxEcalEOverP_1 = cms.double(0.5), #pion rejection
            maxEcalEOverP_2 = cms.double(0.2), #weird events
            maxEeleOverPout = cms.double(0.2),
            maxDPhiIN = cms.double(0.1)
        ),
        electron_protectionsForBadHcal = cms.PSet(
            enableProtections = cms.bool(False),
            full5x5_sigmaIetaIeta = cms.vdouble(0.0106, 0.0387), # EB, EE; 94Xv2 cut-based medium id
            eInvPInv = cms.vdouble(0.184, 0.0721),
            dEta = cms.vdouble(0.0032*2, 0.00632*2), # relax factor 2 to be safer against misalignment
            dPhi = cms.vdouble(0.0547, 0.0394),
        ),

        # New photon selection cuts for CMSSW_700
        photon_MinEt = cms.double(10.),
        photon_combIso = cms.double(10.),
        photon_HoE =  cms.double(0.05),
        photon_SigmaiEtaiEta_barrel = cms.double(0.0125),
        photon_SigmaiEtaiEta_endcap = cms.double(0.034),
        photon_protectionsForBadHcal = cms.PSet(
            enableProtections = cms.bool(False),
            solidConeTrkIsoOffset = cms.double(10.),
            solidConeTrkIsoSlope  = cms.double(0.3),
        ),
        photon_protectionsForJetMET = cms.PSet(
            sumPtTrackIso = cms.double(4.0),
            sumPtTrackIsoSlope = cms.double(0.001)
        )
    ), # PFEGammaFiltersParameters ends
    #----------------------------------

    # Input displaced vertices
    # It is strongly adviced to keep usePFNuclearInteractions = bCorrect                       
                          
    rejectTracks_Bad =  cms.bool(True),
    rejectTracks_Step45 = cms.bool(True),

    usePFNuclearInteractions = cms.bool(True),
    usePFConversions = cms.bool(True),
    usePFDecays = cms.bool(False),

    dptRel_DispVtx = cms.double(10.),

    iCfgCandConnector = cms.PSet(
         bCorrect         =  cms.bool(True), 
         bCalibPrimary    =  cms.bool(True),
         dptRel_PrimaryTrack = cms.double(10.),
         dptRel_MergedTrack = cms.double(5.0),
         ptErrorSecondary = cms.double(1.0),
         nuclCalibFactors =  cms.vdouble(0.8, 0.15, 0.5, 0.5, 0.05)
    ),

    # Treatment of muons : 
    # Expected energy in ECAL and HCAL, and RMS
    muon_HCAL = cms.vdouble(3.0,3.0),
    muon_ECAL = cms.vdouble(0.5,0.5),
    muon_HO = cms.vdouble(0.9,0.9),		

    #----------------------------------
    # For PFMuonAlgo
    #----------------------------------
    PFMuonAlgoParameters = cms.PSet(

        # Muon ID and post cleaning parameters
        maxDPtOPt      = cms.double(1.),
        minTrackerHits = cms.int32(8),
        minPixelHits   = cms.int32(1),
        trackQuality   = cms.string('highPurity'),
        dzPV = cms.double(0.2),
        ptErrorScale   = cms.double(8.),
        minPtForPostCleaning = cms.double(20.),
        eventFactorForCosmics =cms.double(10.),
        metSignificanceForCleaning = cms.double(3.),
        metSignificanceForRejection = cms.double(4.),
        metFactorForCleaning = cms.double(4.),
        eventFractionForCleaning =cms.double(0.5),
        eventFractionForRejection = cms.double(0.8),
        metFactorForRejection =cms.double(4.),
        metFactorForHighEta   = cms.double(25.),
        ptFactorForHighEta   = cms.double(2.),
        metFactorForFakes    = cms.double(4.),
        minMomentumForPunchThrough = cms.double(100.),
        minEnergyForPunchThrough = cms.double(100.),
        punchThroughFactor = cms.double(3.),
        punchThroughMETFactor = cms.double(4.),
        cosmicRejectionDistance = cms.double(1.)
    ),
    #----------------------------------
                             
    # Treatment of potential fake tracks
    # Number of sigmas for fake track detection
    nsigma_TRACK = cms.double(1.0),
    # Absolute pt error to detect fake tracks in the first three iterations
    # dont forget to modify also ptErrorSecondary if you modify this parameter
    pt_Error = cms.double(1.0),
    # Factors to be applied in the four and fifth steps to the pt error
    factors_45 = cms.vdouble(10.,100.),

    #----------------------------------
    # Treatment of tracks in region of bad HCal
    #----------------------------------
    PFBadHcalMitigationParameters = cms.PSet(
        goodTrackDeadHcal_ptErrRel = cms.double(0.2), # trackRef->ptError()/trackRef->pt() < X
        goodTrackDeadHcal_chi2n = cms.double(5),      # trackRef->normalizedChi2() < X
        goodTrackDeadHcal_layers = cms.uint32(4),     # trackRef->hitPattern().trackerLayersWithMeasurement() >= X
        goodTrackDeadHcal_validFr = cms.double(0.5),  # trackRef->validFraction() > X
        goodTrackDeadHcal_dxy = cms.double(0.5),      # [cm] abs(trackRef->dxy(primaryVertex_.position())) < X

        goodPixelTrackDeadHcal_minEta = cms.double(2.3),   # abs(trackRef->eta()) > X
        goodPixelTrackDeadHcal_maxPt  = cms.double(50.),   # trackRef->ptError()/trackRef->pt() < X
        goodPixelTrackDeadHcal_ptErrRel = cms.double(1.0), # trackRef->ptError()/trackRef->pt() < X
        goodPixelTrackDeadHcal_chi2n = cms.double(2),      # trackRef->normalizedChi2() < X
        goodPixelTrackDeadHcal_maxLost3Hit = cms.int32(0), # max missing outer hits for a track with 3 valid pixel layers (can set to -1 to reject all these tracks)
        goodPixelTrackDeadHcal_maxLost4Hit = cms.int32(1), # max missing outer hits for a track with >= 4 valid pixel layers
        goodPixelTrackDeadHcal_dxy = cms.double(0.02),     # [cm] abs(trackRef->dxy(primaryVertex_.position())) < X
        goodPixelTrackDeadHcal_dz  = cms.double(0.05)     # [cm] abs(trackRef->dz(primaryVertex_.position())) < X
    ),
    #----------------------------------

    #----------------------------------
    # Post HF cleaning
    #----------------------------------
    postHFCleaning = cms.bool(False),
    PFHFCleaningParameters = cms.PSet(
        # Clean only objects with pt larger than this value
        minHFCleaningPt = cms.double(5.),
        # Clean only if the initial MET/sqrt(sumet) is larger than this value
        maxSignificance = cms.double(2.5),
        # Clean only if the final MET/sqrt(sumet) is smaller than this value
        minSignificance = cms.double(2.5),
        # Clean only if the significance reduction is larger than this value
        minSignificanceReduction = cms.double(1.4),
        # Clean only if the MET and the to-be-cleaned object satisfy this DeltaPhi * Pt cut
        # (the MET angular resoution is in 1/MET)
        maxDeltaPhiPt = cms.double(7.0),
        # Clean only if the MET relative reduction from the to-be-cleaned object
        # is larger than this value
        minDeltaMet = cms.double(0.4)
    ),
    #----------------------------------

    # Check HF cleaning
    cleanedHF = cms.VInputTag(
                cms.InputTag("particleFlowRecHitHF","Cleaned"),
                cms.InputTag("particleFlowClusterHF","Cleaned")
                ),
  
    # number of sigmas for neutral energy detection
    pf_nsigma_ECAL = cms.double(0.0),
    pf_nsigma_HCAL = cms.double(1.0),

    # ECAL/HCAL PF cluster calibration : take it from global tag ?
    useCalibrationsFromDB = cms.bool(True),
    calibrationsLabel = cms.string(''),

    # calibration parameters for HF:
    calibHF_use = cms.bool(False),
    calibHF_eta_step  = cms.vdouble(0.0,2.90,3.00,3.20,4.20,4.40,4.60,4.80,5.20,5.40),
    calibHF_a_EMonly  = cms.vdouble(0.96945,0.96701,0.76309,0.82268,0.87583,0.89718,0.98674,1.4681,1.4580,1.4580),
    calibHF_b_HADonly = cms.vdouble(1.27541,0.85361,0.86333,0.89091,0.94348,0.94348,0.94370,1.0034,1.0444,1.0444),
    calibHF_a_EMHAD   = cms.vdouble(1.42215,1.00496,0.68961,0.81656,0.98504,0.98504,1.00802,1.0593,1.4576,1.4576),
    calibHF_b_EMHAD   = cms.vdouble(1.27541,0.85361,0.86333,0.89091,0.94348,0.94348,0.94370,1.0034,1.0444,1.0444)

)

from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify(particleFlowTmp,
        electron_protectionsForBadHcal = dict(enableProtections = True),
        photon_protectionsForBadHcal   = dict(enableProtections = True))

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowTmp,photon_MinEt = 1.)
