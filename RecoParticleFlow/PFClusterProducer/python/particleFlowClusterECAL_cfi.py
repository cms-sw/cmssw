import FWCore.ParameterSet.Config as cms


particleFlowClusterECAL = cms.EDProducer("PFClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    #corrections
    applyCrackCorrections = cms.bool(False),
    # PFRecHit collection          
    PFRecHits = cms.InputTag("particleFlowRecHitECAL"),
    PFClustersPS = cms.InputTag('particleFlowClusterPS'), #for EE->PS assoc.
    thresh_Preshower = cms.double(0.0),
    #PFCluster Collection name
    #PFClusterCollectionName =  cms.string("ECAL"),                                
    #----all thresholds are in GeV
    # seed threshold in ECAL barrel 
    thresh_Seed_Barrel = cms.double(0.23),
    thresh_Pt_Seed_Barrel = cms.double(0.00),
    # cell threshold in ECAL barrel 
    thresh_Barrel = cms.double(0.08),
    thresh_Pt_Barrel = cms.double(0.00),
    # cleaning threshold and minimum S4/S1 fraction in ECAL barrel
    thresh_Clean_Barrel = cms.double(4.0),
    minS4S1_Clean_Barrel = cms.vdouble(0.04, -0.024),
    # double spike cleaning (barrel)
    thresh_DoubleSpike_Barrel = cms.double(10.),
    minS6S2_DoubleSpike_Barrel = cms.double(0.04),
    # seed threshold in ECAL endcap 
    thresh_Seed_Endcap = cms.double(0.6),
    thresh_Pt_Seed_Endcap = cms.double(0.15),
    # cell threshold in ECAL endcap 
    thresh_Endcap = cms.double(0.3),
    thresh_Pt_Endcap = cms.double(0.00),
    # cleaning threshold and minimum S4/S1 fraction in ECAL endcap
    thresh_Clean_Endcap = cms.double(15.0),
    minS4S1_Clean_Endcap = cms.vdouble(0.02, -0.0125),
    # double spike cleaning (endcap)
    thresh_DoubleSpike_Endcap = cms.double(1E9),
    minS6S2_DoubleSpike_Endcap = cms.double(-1.),
    # thresh_Clean_Endcap = cms.double(1E5),
    #minS4S1_Clean_Endcap = cms.vdouble(0.04, -0.025),
    # n neighbours in ECAL 
    nNeighbours = cms.int32(8),
    # sigma of the shower in ECAL 
    showerSigma = cms.double(1.5),
    # n crystals for position calculation in ECAL
    posCalcNCrystal = cms.int32(9),
    # use cells with common corner to build topo-clusters
    useCornerCells = cms.bool(True),
    # enable cleaning of RBX and HPD (HCAL only);             
    cleanRBXandHPDs = cms.bool(False),
    PositionCalcType = cms.string('EGPositionCalc'),
    # e/gamma position calc config  
    PositionCalcConfig = cms.PSet( T0_barl = cms.double(7.4),
                                   T0_endc = cms.double(3.1),
                                   T0_endcPresh = cms.double(1.2),
                                   LogWeighted = cms.bool(True),
                                   W0 = cms.double(4.2),
                                   X0 = cms.double(0.89)
                                   ),
    # depth correction for ECAL clusters:
    #   0: no depth correction
    #   1: electrons/photons - depth correction is proportionnal to E
    #   2: hadrons - depth correction is fixed
    depthCor_Mode = cms.int32(1),
    # in mode 1, depth correction = A *( B + log(E) )
    # in mode 2, depth correction = A 
    depthCor_A = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    # under the preshower, the depth is smaller, but the material is 
    # the same
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B_preshower = cms.double(4.0)
)


