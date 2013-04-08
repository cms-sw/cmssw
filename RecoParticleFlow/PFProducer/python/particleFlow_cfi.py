import FWCore.ParameterSet.Config as cms

particleFlowTmp = cms.EDProducer("PFProducer",

    # PF Blocks label
    blocks = cms.InputTag("particleFlowBlock"),

    # reco::muons label and Post Muon cleaning
    muons = cms.InputTag("muons1stStep"),
    postMuonCleaning = cms.bool(True),

    # Vertices label
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
    useVerticesForNeutral = cms.bool(True),

    # Algorithm type ?
    algoType = cms.uint32(0),

    # Verbose and debug flags
    verbose = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),

    # Use HO clusters in PF hadron reconstruction
    useHO = cms.bool(True),                                 

    # Use electron identification in PFAlgo
    usePFElectrons = cms.bool(True),
    pf_electron_output_col=cms.string('electrons'),
    pf_electronID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_PfElectrons23Jan_IntToFloat.txt'),

    # Use Photon identification in PFAlgo (for now this has NO impact, algo is swicthed off hard-coded
    usePFPhotons = cms.bool(True),
    usePhotonReg=cms.bool(False),
    useRegressionFromDB=cms.bool(True),                                 
    pf_convID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_pfConversionAug0411.txt'),        
    pf_conv_mvaCut=cms.double(0.0),                                 
    pf_locC_mvaWeightFile=cms.string('RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFClusterLCorr_14Dec2011.root'),
    pf_GlobC_mvaWeightFile=cms.string('RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFGlobalCorr_14Dec2011.root'),
    pf_Res_mvaWeightFile=cms.string('RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFRes_14Dec2011.root'),
    X0_Map=cms.string('RecoParticleFlow/PFProducer/data/allX0histos.root'),
    sumPtTrackIsoForPhoton=cms.double(2.0),
    sumPtTrackIsoSlopeForPhoton=cms.double(0.001),

                              
    pf_electron_mvaCut = cms.double(-0.1),
    # apply the crack corrections                             
    pf_electronID_crackCorrection = cms.bool(False),
    usePFSCEleCalib = cms.bool(True),
                              #new corrections  #MM /*
    calibPFSCEle_Fbrem_barrel = cms.vdouble(0.6, 6,                                                 #Range of non constant correction
                                            -0.0255975, 0.0576727, 0.975442, -0.000546394, 1.26147, #standard parameters
                                            25,                                                     #pt value for switch to low pt corrections
                                            -0.02025, 0.04537, 0.9728, -0.0008962, 1.172),          # low pt parameters
    calibPFSCEle_Fbrem_endcap = cms.vdouble(0.9, 6.5,                                               #Range of non constant correction
                                            -0.0692932, 0.101776, 0.995338, -0.00236548, 0.874998,  #standard parameters eta < switch value
                                            1.653,                                                  #eta value for correction switch
                                            -0.0750184, 0.147000, 0.923165, 0.000474665, 1.10782),  #standard parameters eta > switch value
    calibPFSCEle_barrel = cms.vdouble(1.004, -1.536, 22.88, -1.467,  #standard
                                      0.3555, 0.6227, 14.65, 2051,   #parameters
                                      25,                            #pt value for switch to low pt corrections
                                      0.9932, -0.5444, 0, 0.5438,    #low pt
                                      0.7109, 7.645, 0.2904, 0),     #parameters
    calibPFSCEle_endcap = cms.vdouble(1.153, -16.5975, 5.668,
                                      -0.1772, 16.22, 7.326,
                                      0.0483, -4.068, 9.406),
                              #old corrections #MM */
#    calibPFSCEle_barrel = cms.vdouble(1.0326,-13.71,339.72,0.4862,0.00182,0.36445,1.411,1.0206,0.0059162,-5.14434e-05,1.42516e-07),
#    calibPFSCEle_endcap = cms.vdouble(0.9995,-12.313,2.8784,-1.057e-04,10.282,3.059,1.3502e-03,-2.2185,3.4206),

    useEGammaSupercluster =  cms.bool(True),
    sumEtEcalIsoForEgammaSC_barrel = cms.double(1.),
    sumEtEcalIsoForEgammaSC_endcap = cms.double(2.),
    coneEcalIsoForEgammaSC = cms.double(0.3),
    sumPtTrackIsoForEgammaSC_barrel = cms.double(4.),
    sumPtTrackIsoForEgammaSC_endcap = cms.double(4.),
    nTrackIsoForEgammaSC = cms.uint32(2),                          
    coneTrackIsoForEgammaSC = cms.double(0.3),
    useEGammaElectrons = cms.bool(True),                                 
    egammaElectrons = cms.InputTag('mvaElectrons'),                              

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
    eventFractionForCleaning =cms.double(0.8),
    eventFractionForRejection = cms.double(0.8),
    metFactorForRejection =cms.double(4.),
    metFactorForHighEta   = cms.double(25.),
    ptFactorForHighEta   = cms.double(2.),
    metFactorForFakes    = cms.double(4.),
    minMomentumForPunchThrough = cms.double(100.),
    minEnergyForPunchThrough = cms.double(100.),
    punchThroughFactor = cms.double(3.),                             
    punchThroughMETFactor = cms.double(4.),                             
    cosmicRejectionDistance = cms.double(1.),
                                 
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

    # ECAL/HCAL PF cluster calibration : take it from global tag ?
    useCalibrationsFromDB = cms.bool(True),

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
 
#    toRead = cms.untracked.vstring("PFfa_BARREL",
#                                   "PFfa_ENDCAP",
#                                   "PFfb_BARREL",
#                                   "PFfb_ENDCAP",
#                                   "PFfc_BARREL",
#                                   "PFfc_ENDCAP",
#                                   "PFfaEta_BARREL",
#                                   "PFfaEta_ENDCAP",
#                                   "PFfbEta_BARREL",
#                                   "PFfbEta_ENDCAP") # same strings as fType

)



