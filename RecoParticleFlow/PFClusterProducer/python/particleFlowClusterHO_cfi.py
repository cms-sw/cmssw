import FWCore.ParameterSet.Config as cms


particleFlowClusterHO = cms.EDProducer("PFClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFRecHit collection          
    PFRecHits = cms.InputTag("particleFlowRecHitHO"),

    # seed threshold in HO barrel 
    thresh_Seed_Barrel = cms.double(1.0),
    thresh_Pt_Seed_Barrel = cms.double(0.0),
    # cell threshold in HCAL barrel 
    thresh_Barrel = cms.double(0.5),
    thresh_Pt_Barrel = cms.double(0.0),
    # cleaning threshold and minimum S4/S1 fraction in HCAL barrel
    # thresh_Clean_Barrel = cms.double(35.0),
    thresh_Clean_Barrel = cms.double(1E5),
    minS4S1_Clean_Barrel = cms.vdouble(0.032, -0.045),
    # double spike cleaning (barrel)
    thresh_DoubleSpike_Barrel = cms.double(1E9),
    minS6S2_DoubleSpike_Barrel = cms.double(-1.),
    # seed threshold in HCAL endcap 
    thresh_Seed_Endcap = cms.double(3.1),
    thresh_Pt_Seed_Endcap = cms.double(0.0),
    # cell threshold in HCAL endcap
    thresh_Endcap = cms.double(1.0),    
    thresh_Pt_Endcap = cms.double(0.0),
    # cleaning threshold and minimum S4/S1 fraction in HCAL barrel
    #thresh_Clean_Endcap = cms.double(50.0),
    thresh_Clean_Endcap = cms.double(1E5),
    minS4S1_Clean_Endcap = cms.vdouble(0.032, -0.045),
    
    # double spike cleaning (endcap)
    thresh_DoubleSpike_Endcap = cms.double(1E9),
    minS6S2_DoubleSpike_Endcap = cms.double(-1.),
    #----HCAL options
    # n neighbours in HCAL 
    nNeighbours = cms.int32(4),
    # sigma of the shower in HCAL     
    showerSigma = cms.double(10.0),
    # use cells with common corner to build topo-clusters
    useCornerCells = cms.bool(True),
    # enable cleaning of RBX and HPD (HCAL only);
    
    cleanRBXandHPDs = cms.bool(False),
    # n crystals for position calculation in HCAL
    posCalcNCrystal = cms.int32(5), 
    #----depth correction
    # depth correction for ECAL clusters:
    #   0: no depth correction
    #   1: electrons/photons - depth correction is proportionnal to E
    #   2: hadrons - depth correction is fixed
    depthCor_Mode = cms.int32(0),
    # in mode 1, depth correction = A *( B + log(E) )
    # in mode 2, depth correction = A 
    depthCor_A = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    # under the preshower, the depth is smaller, but the material is 
    # the same
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B_preshower = cms.double(4.0)
                                           
)


