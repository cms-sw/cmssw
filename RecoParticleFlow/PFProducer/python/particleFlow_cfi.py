import FWCore.ParameterSet.Config as cms

particleFlow = cms.EDProducer("PFProducer",

    # PF Blocks label
    blocks = cms.InputTag("particleFlowBlock"),

    # Algorithm type ?
    algoType = cms.uint32(0),

    # Verbose and debug flags
    verbose = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),

    # Use electron identification in PFAlgo
    usePFElectrons = cms.bool(False),
    pf_electronID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_finalID_hzz-pions.txt'),
    final_chi2cut_bremps = cms.double(25.0),
    final_chi2cut_bremecal = cms.double(25.0),
    final_chi2cut_bremhcal = cms.double(25.0),
    final_chi2cut_gsfps = cms.double(100.0),
    final_chi2cut_gsfecal = cms.double(900.0),
    final_chi2cut_gsfhcal = cms.double(100.0),
    pf_electron_mvaCut = cms.double(-0.4),

    # Use photon conversion identification in PFAlgo
    usePFConversions = cms.bool(False),

    # Merged photons
    pf_mergedPhotons_PSCut = cms.double(0.001),
    pf_mergedPhotons_mvaCut = cms.double(0.5),
    pf_mergedPhotons_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_MLP.weights.txt'),

    # number of sigmas for neutral energy detection
    pf_nsigma_ECAL = cms.double(3.0),
    pf_nsigma_HCAL = cms.double(1.0),

    # Cluster recovery                              
    pf_clusterRecovery = cms.bool(False),

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

    # Brand-new cluster calibration !
    # Use this new calibration ?
    pf_newCalib = cms.bool(False),
    # Apply corrections?
    pfcluster_doCorrection = cms.uint32(1),
    # Bulk correction parameters
    pfcluster_globalP0 = cms.double(-2.315),
    pfcluster_globalP1 = cms.double(1.05),
    # Low energy correction parameters
    pfcluster_lowEP0 = cms.double(5.906466e-01),
    pfcluster_lowEP1 = cms.double(4.608835e-01),
    pfcluster_allowNegative     = cms.uint32(0),
    pfcluster_maxEToCorrect = cms.double(-1.0),
    pfcluster_doEtaCorrection = cms.uint32(1),
    pfcluster_ecalECut = cms.double(0.0),
    pfcluster_hcalECut = cms.double(0.0),
    pfcluster_barrelEndcapEtaDiv = cms.double(1.4),
    #Use hand fitted parameters specified below
    ecalHcalEcalBarrel = cms.vdouble(0.0,     0.0,    1.15,   0.90,   -0.035,         1.1),
    ecalHcalEcalEndcap = cms.vdouble(0.280,   5.0,    1.10,   0.40,   -0.020,         1.1),
    ecalHcalHcalBarrel = cms.vdouble(0.260,   5.0,    1.15,   0.30,   -0.020,         1.1),
    ecalHcalHcalEndcap = cms.vdouble(0.260,   5.0,    1.10,   0.30,   -0.020,         1.1),
    # Fitted parameters for eta correction
    pfcluster_etaCorrection = cms.vdouble( 0.0127, 0.0479,  1.40,    -0.50,   1.00,    1.50,   -0.0132, 0.009507  )

)



