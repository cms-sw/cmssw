import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 
_densityBasedCleaner = cms.PSet(
    algoName = cms.string("PandoraIsolatedSpikeKiller"),
    hit_search_radius = cms.double(6.0),
    hit_search_length = cms.double(20.0),
    weight_power = cms.double(2.0),
    weight_cut = cms.double(0.75)
    )

#seeding
_localmaxseeds_HGCEE = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HGC_ECAL"),
              #seeding threshold converted to GeV from keV
              seedingThreshold = cms.double(1e-6*1.75*55.1),
              seedingThresholdPt = cms.double(0.0)
              ),
     cms.PSet( detector = cms.string("HGC_HCALF"),
              #seeding threshold converted to GeV from keV
              seedingThreshold = cms.double(1e-6*1.0*85.0),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HGC_HCALB"),
              #seeding threshold converted to GeV from keV
              seedingThreshold = cms.double(1e-6*1.0*1498.4),
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(8)
)

optimized_w0s = cms.vdouble(4.0,4.0,4.0,4.0,4.0,4.0,  #1-6 are all the same
                            2.55,
                            2.9,
                            2.45,
                            2.75,
                            2.35,
                            2.55,
                            2.2,
                            2.35,
                            2.0,
                            2.2,
                            1.9,
                            2.05,
                            1.75,
                            1.9,
                            1.7,
                            1.8,
                            3.0,
                            4.0,4.0,4.0,4.0,4.0,4.0,4.0)

_positionCalcHGCEE_onelayer = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1), 
    logWeightDenominator = cms.double(1e-6*0.25*55.1), # use ADC value 0.25*MIP
    minAllowedNormalization = cms.double(1e-9),
    # ECAL specific
    # from layer 1 to 30, these w0s are for w_i/w_tot style log-weights
    w0PerLayer = optimized_w0s
    )

_positionCalcHGCEE_pca = cms.PSet(
    algoName = cms.string("Cluster3DPCACalculator"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(1e-6*1.0*55.1), # use 1 MIP
    minAllowedNormalization = cms.double(1e-9),
    # ECAL specific
    # from layer 1 to 30, these w0s are for w_i/w_tot style log-weights
    w0PerLayer = optimized_w0s,
    logWeightScale = cms.double(250.0)
    )

weight_vec_ee_electrons = [0.080]
weight_vec_ee_electrons.extend([0.620 for x in range(10)])
weight_vec_ee_electrons.extend([0.809 for x in range(10)])
weight_vec_ee_electrons.extend([1.239 for x in range(9)])
weight_vec_ee_hadrons   = [0.0179]
weight_vec_ee_hadrons.extend([0.0105 for x in range(10)])
weight_vec_ee_hadrons.extend([0.0096 for x in range(10)])
weight_vec_ee_hadrons.extend([0.0169 for x in range(9)])

weight_vec_hef          = [0.0464]
weight_vec_hef.extend([0.0474 for x in range(11)])
weight_vec_heb          = [0.1215 for x in range(12)]

#lambda based hadron weights
weight_vec_ee_hadrons_lambda = [0.0100]
weight_vec_ee_hadrons_lambda.extend([0.0360 for x in range(10)])
weight_vec_ee_hadrons_lambda.extend([0.0320 for x in range(10)])
weight_vec_ee_hadrons_lambda.extend([0.0560 for x in range(9)])

weight_vec_hef_lambda = [0.338]
weight_vec_hef_lambda.extend([0.273 for x in range(11)])
weight_vec_heb_lambda = [0.476 for x in range(12)]

_HGCEE_EMEnergyCalibrator = cms.PSet(
    algoName = cms.string("HGCAllLayersEnergyCalibrator"),    
    MipValueInGeV_ee = cms.double(55.1*1e-6),
    MipValueInGeV_hef = cms.double(85.0*1e-6),
    MipValueInGeV_heb = cms.double(1498.4*1e-6),
    #EM energy calibrations
    weights_ee = cms.vdouble(weight_vec_ee_electrons),
    weights_hef = cms.vdouble(weight_vec_hef),
    weights_heb = cms.vdouble(weight_vec_heb),
    #EM calibration
    effMip_to_InverseGeV_a = cms.double(80.0837),
    effMip_to_InverseGeV_b = cms.double(-107.229),
    effMip_to_InverseGeV_c = cms.double(0.0472817),    
    effMip_to_InverseGeV_d = cms.double(-0.266294),    
    effMip_to_InverseGeV_e = cms.double(0.34684),
    # hadron calibration
    ee_had_emscale_slope = cms.double(0.2339),
    ee_had_emscale_offset = cms.double(0.1778),
    hef_had_emscale_slope = cms.double(0.1828),
    hef_had_emscale_offset = cms.double(0.9601),
    heb_had_emscale_slope = cms.double(0.2471),
    heb_had_emscale_offset = cms.double(1.4563),
    he_had_correction = cms.double(1.29208),
    heb_had_correction = cms.double(1.0535),
    pion_energy_slope = cms.double(1.0544),
    pion_energy_offset = cms.double(1.7946),
    had_residual = cms.vdouble(0.434875,-0.3441,0.0631),
    #overburden / driver
    hgcOverburdenParamFile = cms.FileInPath('RecoParticleFlow/PFClusterProducer/data/HGCMaterialOverburden.root'),
    isEMCalibration = cms.bool(True)
)

_HGCEE_HADEnergyCalibrator = _HGCEE_EMEnergyCalibrator.clone()
_HGCEE_HADEnergyCalibrator.weights_ee = cms.vdouble(weight_vec_ee_hadrons_lambda)
_HGCEE_HADEnergyCalibrator.weights_hef = cms.vdouble(weight_vec_hef_lambda)
_HGCEE_HADEnergyCalibrator.weights_heb = cms.vdouble(weight_vec_heb_lambda)
_HGCEE_HADEnergyCalibrator.isEMCalibration = cms.bool(False)

#print _HGCEE_HADEnergyCalibrator.weights_ee, len(_HGCEE_HADEnergyCalibrator.weights_ee)
#print _HGCEE_HADEnergyCalibrator.weights_he, len(_HGCEE_HADEnergyCalibrator.weights_he)

_fromScratchHGCClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("HGCClusterizer"), 
    thresholdsByDetector = cms.VPSet( ),
    positionCalcInLayer = _positionCalcHGCEE_onelayer,
    positionCalcPCA = _positionCalcHGCEE_pca,
    hgcalGeometryNames = cms.PSet( HGC_ECAL  = cms.string('HGCalEESensitive'),
                                   HGC_HCALF = cms.string('HGCalHESiliconSensitive'),
                                   HGC_HCALB = cms.string('HGCalHEScintillatorSensitive') ),
    # this parameterization gives a value for the radius in millimeters using e^(const. + slope*<layer>)
    emShowerParameterization = cms.PSet( HGC_ECAL_constant = cms.double(2.11873),
                                         HGC_ECAL_slope = cms.double(0.0796206),
                                         HGC_ECAL_max_radius = cms.double(60.0) ),
    moliereRadii = cms.PSet( HGC_ECAL = cms.double(2.9),
                             HGC_HCALF = cms.double(7.0),
                             HGC_HCALB = cms.double(8.0) ), #cm
    radiationLengths = cms.PSet( HGC_ECAL = cms.vdouble(1.0),
                                 HGC_HCAL = cms.vdouble(1.0) ),
    interactionLengths = cms.PSet( HGC_ECAL = cms.vdouble(1.0),
                                   HGC_HCAL = cms.vdouble(1.0) ), 
    useTrackAssistedClustering = cms.bool(True),
    trackAssistedClustering = cms.PSet( 
        inputTracks = cms.InputTag("generalTracks"),
        cleaningCriteriaPerIter = cms.vdouble(1.0),
        stoppingTolerance = cms.double(1.0), #Nsigma to stop cluster growth
        stopAtFirstClusterEncountered = cms.bool(False),
        expectedHadronicResolution = cms.PSet( stochastic = cms.double(70.0),
                                               noise = cms.double(0.0),
                                               constant = cms.double(1.0) ), 
        #cluster afterburner
        useAfterburner = cms.bool(True),
        minConeAngle = cms.double(0.1), # radians
        maxConeAngle = cms.double(0.5), # radians
        maxConeDepth = cms.double(60), #cm
        maxClusterAngleToTrack = cms.double(0.20), # radians
        minECALLayerToCone = cms.uint32(11), # ~9.5 radiation lengths
        ),
    
    #energy calibrations
    emEnergyCalibration  = _HGCEE_EMEnergyCalibrator,
    hadEnergyCalibration = _HGCEE_HADEnergyCalibrator
    
)

#weights for layers from P.Silva (24 October 2014)
## this is for V5!
weight_vec = [0.080]
weight_vec.extend([0.62 for x in range(10)])
weight_vec.extend([0.81 for x in range(10)])
weight_vec.extend([1.19 for x in range(9)])

# MIP effective to 1.0/GeV (from fit to data of P. Silva)
#f(x) = a/(1-exp(-bx - c))
# x = cosh(eta)
# a = 82.8
# b = 1e6
# c = 1e6

_HGCEE_ElectronEnergy = cms.PSet(
    algoName = cms.string("HGCEEElectronEnergyCalibrator"),
    weights = cms.vdouble(weight_vec),
    effMip_to_InverseGeV_a = cms.double(82.8),
    effMip_to_InverseGeV_b = cms.double(1e6),
    effMip_to_InverseGeV_c = cms.double(1e6),
    MipValueInGeV = cms.double(55.1*1e-6)
)

particleFlowClusterHGCEE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCEE"),
    recHitCleaners = cms.VPSet(_densityBasedCleaner),
    seedFinder = _localmaxseeds_HGCEE,
    initialClusteringStep = _fromScratchHGCClusterizer_HGCEE,
    pfClusterBuilder = cms.PSet( ), #_arborClusterizer_HGCEE,
    positionReCalc = cms.PSet( ), #_simplePosCalcHGCEE,
    energyCorrector = cms.PSet( ) #_HGCEE_EMEnergyCalibrator
)

