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
    usePFElectrons = cms.bool(True),
    pf_electron_output_col=cms.string('electrons'),
    pf_electronID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_PfElectrons23Jan.txt'),
                              
    final_chi2cut_bremps = cms.double(0.0),
    final_chi2cut_bremecal = cms.double(0.0),
    final_chi2cut_bremhcal = cms.double(0.0),
    final_chi2cut_gsfps = cms.double(0.0),
    final_chi2cut_gsfecal = cms.double(0.0),
    final_chi2cut_gsfhcal = cms.double(0.0),
    pf_electron_mvaCut = cms.double(-0.1),

    # Use photon conversion identification in PFAlgo
    usePFConversions = cms.bool(False),

    # Treatment of muons : 
    # Expected energy in ECAL and HCAL, and RMS
    muon_HCAL = cms.vdouble(3.0,3.0),
    muon_ECAL = cms.vdouble(0.5,0.5),

    # Treatment of potential fake tracks
    # Number of sigmas for fake track detection
    nsigma_TRACK = cms.double(3.0),
    # Absolute pt error to detect fake tracks in the first three iterations
    pt_Error = cms.double(1.0),
    # Factors to be applied in the four and fifth steps to the pt error
    factors_45 = cms.vdouble(1.,1.),

    # Merged photons
    pf_mergedPhotons_PSCut = cms.double(0.001),
    pf_mergedPhotons_mvaCut = cms.double(0.5),
    pf_mergedPhotons_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_MLP.weights.txt'),

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
    pfcluster_etaCorrection = cms.vdouble(1.01,   -1.02e-02,   5.17e-02,      0.563,     -0.425,     0.110)

)



