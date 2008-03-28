import FWCore.ParameterSet.Config as cms

particleFlowClusterECAL = cms.EDProducer("PFClusterProducer",
    # seed threshold in ECAL endcap 
    thresh_Seed_Endcap = cms.double(0.8),
    # verbosity 
    verbose = cms.untracked.bool(False),
    # sigma of the shower in ECAL 
    showerSigma = cms.double(5.0),
    # seed threshold in ECAL barrel 
    thresh_Seed_Barrel = cms.double(0.23),
    # depth correction for ECAL clusters:
    #   0: no depth correction
    #   1: electrons/photons - depth correction is proportionnal to E
    #   2: hadrons - depth correction is fixed
    depthCor_Mode = cms.int32(1),
    # n crystals for position calculation in ECAL
    posCalcNCrystal = cms.int32(9),
    depthCor_B_preshower = cms.double(4.0),
    # n neighbours in ECAL 
    nNeighbours = cms.int32(8),
    PFRecHits = cms.InputTag("particleFlowRecHitECAL"),
    # all thresholds are in GeV
    # cell threshold in ECAL barrel 
    thresh_Barrel = cms.double(0.08),
    # under the preshower, the depth is smaller, but the material is 
    # the same
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    # in mode 1, depth correction = A *( B + log(E) )
    # in mode 2, depth correction = A 
    depthCor_A = cms.double(0.89),
    # cell threshold in ECAL endcap 
    thresh_Endcap = cms.double(0.3),
    # p1 for position calculation in ECAL 
    posCalcP1 = cms.double(-1.0)
)


