import FWCore.ParameterSet.Config as cms

particleFlowClusterHCAL = cms.EDProducer("PFClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFRecHit collection                                  
    PFRecHits = cms.InputTag("particleFlowRecHitHCAL"),
    #----all thresholds are in GeV
    # seed threshold in HCAL barrel 
    thresh_Seed_Barrel = cms.double(0.8),
    thresh_Pt_Seed_Barrel = cms.double(0.0),
    # cell threshold in HCAL barrel 
    thresh_Barrel = cms.double(0.8),
    thresh_Pt_Barrel = cms.double(0.0),
    # seed threshold in HCAL endcap 
    thresh_Seed_Endcap = cms.double(1.1),
    thresh_Pt_Seed_Endcap = cms.double(0.0),
    # cell threshold in HCAL endcap
    thresh_Endcap = cms.double(0.8),    
    thresh_Pt_Endcap = cms.double(0.0),    
    #----HCAL options
    # n neighbours in HCAL 
    nNeighbours = cms.int32(4),
    # sigma of the shower in HCAL     
    showerSigma = cms.double(10.0),
    # use cells with common corner to build topo-clusters
    useCornerCells = cms.bool(True),
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


