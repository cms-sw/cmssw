import FWCore.ParameterSet.Config as cms

particleFlowClusterPS = cms.EDProducer("PFClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFRecHit collection
    PFRecHits = cms.InputTag("particleFlowRecHitPS"),
    #PFCluster Collection name
    #PFClusterCollectionName =  cms.string("PS"),
    #----all thresholds are in GeV
    # seed threshold in preshower
    thresh_Seed_Barrel = cms.double(1.2e-4),
    thresh_Pt_Seed_Barrel = cms.double(0.0),
    # cell threshold in preshower
    thresh_Barrel = cms.double(6e-05),                                       
    thresh_Pt_Barrel = cms.double(0.0),                                       
    # for the preshower, endcap values are not used
    thresh_Seed_Endcap = cms.double(1.2e-4),
    thresh_Pt_Seed_Endcap = cms.double(0.0),
    # for the preshower, endcap values are not used 
    thresh_Endcap = cms.double(6e-05),
    thresh_Pt_Endcap = cms.double(0.0),
    # n neighbours in PS 
    nNeighbours = cms.int32(8),                                       
    # sigma of the shower in preshower     
    showerSigma = cms.double(0.2),
    # n crystals for position calculation in PS
    posCalcNCrystal = cms.int32(-1),
    # use cells with common corner to build topo-clusters
    useCornerCells = cms.bool(False),
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


