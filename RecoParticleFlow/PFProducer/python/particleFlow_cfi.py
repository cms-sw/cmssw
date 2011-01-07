import FWCore.ParameterSet.Config as cms

particleFlow = cms.EDProducer("PFProducer",

    # PF Blocks label
    blocks = cms.InputTag("particleFlowBlock"),

    # reco::muons label and Post Muon cleaning
    muons = cms.InputTag("muons"),
    postMuonCleaning = cms.bool(True),

    # Vertices label
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
    useVerticesForNeutral = cms.bool(True),

    # Algorithm type ?
    algoType = cms.uint32(0),

    # Verbose and debug flags
    verbose = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),

    # Use electron identification in PFAlgo
    usePFElectrons = cms.bool(True),
    pf_electron_output_col=cms.string('electrons'),
    pf_electronID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_PfElectrons23Jan_IntToFloat.txt'),
                              
    pf_electron_mvaCut = cms.double(-0.1),
    # apply the crack corrections                             
    pf_electronID_crackCorrection = cms.bool(False),
    usePFSCEleCalib = cms.bool(True),                        
    calibPFSCEle_barrel = cms.vdouble(1.0326,-13.71,339.72,0.4862,0.00182,0.36445,1.411,1.0206,0.0059162,-5.14434e-05,1.42516e-07),
    calibPFSCEle_endcap = cms.vdouble(0.9995,-12.313,2.8784,-1.057e-04,10.282,3.059,1.3502e-03,-2.2185,3.4206),

    useEGammaSupercluster =  cms.bool(True),
    sumEtEcalIsoForEgammaSC_barrel = cms.double(1.),
    sumEtEcalIsoForEgammaSC_endcap = cms.double(2.),
    coneEcalIsoForEgammaSC = cms.double(0.3),
    sumPtTrackIsoForEgammaSC_barrel = cms.double(4.),
    sumPtTrackIsoForEgammaSC_endcap = cms.double(4.),
    nTrackIsoForEgammaSC = cms.uint32(2),                          
    coneTrackIsoForEgammaSC = cms.double(0.3),
    useEGammaElectrons = cms.bool(False),
    egammaElectrons = cms.InputTag(''),                              

    # Input displaced vertices 
                              
    rejectTracks_Bad =  cms.bool(True),
    rejectTracks_Step45 = cms.bool(True),

    usePFNuclearInteractions = cms.bool(True),
    usePFConversions = cms.bool(False),
    usePFDecays = cms.bool(False),

    dptRel_DispVtx = cms.double(10.),

    iCfgCandConnector = cms.PSet(
    
	 bCorrect         =  cms.bool(True), 
	 bCalibPrimary    =  cms.bool(True),
	 nuclCalibFactors =  cms.vdouble(0.8, 0.15, 0.5, 0.5, 0.05),
         dptRel_PrimaryTrack = cms.double(10.),
         dptRel_MergedTrack = cms.double(5.0),
         ptErrorSecondary = cms.double(1.0)
    ),

    

    # Treatment of muons : 
    # Expected energy in ECAL and HCAL, and RMS
    muon_HCAL = cms.vdouble(3.0,3.0),
    muon_ECAL = cms.vdouble(0.5,0.5),

    # Use PF muon momentum assigment instead of default reco muon one
    usePFMuonMomAssign = cms.bool(False),

    # Treatment of potential fake tracks
    # Number of sigmas for fake track detection
    nsigma_TRACK = cms.double(1.0),
    # Absolute pt error to detect fake tracks in the first three iterations
    # dont forget to modify also ptErrorSecondary if you modify this parameter
    pt_Error = cms.double(1.0),
    # Factors to be applied in the four and fifth steps to the pt error
    factors_45 = cms.vdouble(10.,100.),

    # Post HF cleaning
    postHFCleaning = cms.bool(False),
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
    minDeltaMet = cms.double(0.4),

    # Check HF cleaning
    cleanedHF = cms.VInputTag(
                cms.InputTag("particleFlowRecHitHCAL","Cleaned"),
                cms.InputTag("particleFlowClusterHFHAD","Cleaned"),
                cms.InputTag("particleFlowClusterHFEM","Cleaned")
                ),
    
    # number of sigmas for neutral energy detection
    pf_nsigma_ECAL = cms.double(0.0),
    pf_nsigma_HCAL = cms.double(1.0),

    # Naive cluster calibration
    # ECAL alone
    pf_calib_ECAL_offset = cms.double(0.0),
    pf_calib_ECAL_slope = cms.double(1.0),
    # HCAL alone
    pf_calib_HCAL_slope = cms.double(2.17),
    pf_calib_HCAL_offset = cms.double(1.73),
    pf_calib_HCAL_damping = cms.double(2.49),
    # ECAL + HCAL 
    pf_calib_ECAL_HCAL_hslope = cms.double(1.06),
    pf_calib_ECAL_HCAL_eslope = cms.double(1.05),
    pf_calib_ECAL_HCAL_offset = cms.double(6.11),

    # ECAL/HCAL cluster calibration !
    # Colin = 0; Jamie = 1; Newest = 2.
    pf_newCalib = cms.uint32(2),
   # Apply corrections?
    pfcluster_doCorrection = cms.uint32(1),
    # Bulk correction parameters
    pfcluster_globalP0 = cms.double(-2.315),                              
    pfcluster_globalP1 = cms.double(1.01),
    # Low energy correction parameters
    pfcluster_lowEP0 = cms.double(3.249189e-01),
    pfcluster_lowEP1 = cms.double(7.907990e-01),
    pfcluster_allowNegative     = cms.uint32(0),
    pfcluster_doEtaCorrection = cms.uint32(1),
    pfcluster_barrelEndcapEtaDiv = cms.double(1.4),

    #Use hand fitted parameters specified below
    #P1 adjusts the height of the peak
    ecalHcalEcalBarrel = cms.vdouble(0.67,    3.0,    1.15,    0.90,  -0.060,    1.4),
    ecalHcalEcalEndcap = cms.vdouble(0.46,    3.0,    1.10,    0.40,   -0.020,    1.4),
    ecalHcalHcalBarrel = cms.vdouble(0.46,    3.0,    1.15,    0.30,   -0.020,    1.4),
    ecalHcalHcalEndcap = cms.vdouble(0.460,    3.0,    1.10,   0.30,  -0.02,    1.4),
    pfcluster_etaCorrection = cms.vdouble(1.01,   -1.02e-02,   5.17e-02,      0.563,     -0.425,     0.110),

    # calibration parameters for HF:
    calibHF_use = cms.bool(False),
    calibHF_eta_step  = cms.vdouble(0.0,2.90,3.00,3.20,4.20,4.40,4.60,4.80,5.20,5.40),
#    calibHF_a_EMonly  = cms.vdouble(10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00),
#    calibHF_b_HADonly = cms.vdouble(10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00),
#    calibHF_a_EMHAD   = cms.vdouble(10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00),
#    calibHF_b_EMHAD   = cms.vdouble(10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00,10.00)
    calibHF_a_EMonly  = cms.vdouble(0.96945,0.96701,0.76309,0.82268,0.87583,0.89718,0.98674,1.4681,1.4580,1.4580),
    calibHF_b_HADonly = cms.vdouble(1.27541,0.85361,0.86333,0.89091,0.94348,0.94348,0.94370,1.0034,1.0444,1.0444),
    calibHF_a_EMHAD   = cms.vdouble(1.42215,1.00496,0.68961,0.81656,0.98504,0.98504,1.00802,1.0593,1.4576,1.4576),
    calibHF_b_EMHAD   = cms.vdouble(1.27541,0.85361,0.86333,0.89091,0.94348,0.94348,0.94370,1.0034,1.0444,1.0444)
)



