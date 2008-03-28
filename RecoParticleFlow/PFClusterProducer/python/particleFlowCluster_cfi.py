import FWCore.ParameterSet.Config as cms

particleFlowCluster = cms.EDProducer("PFClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # perform clustering in preshower ?
    process_PS = cms.untracked.bool(True),
    ecalRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    # clustering in ECAL ?
    clustering_Ecal = cms.untracked.bool(True),
    depthCor_B = cms.double(7.4),
    # cell threshold in HCAL barrel 
    thresh_Hcal_Barrel = cms.double(0.8),
    # in mode 1, depth correction = A *( B + log(E) )
    # in mode 2, depth correction = A 
    depthCor_A = cms.double(0.89),
    # p1 for position calculation in PS 
    posCalcP1_PS = cms.double(0.0),
    # if yes, rechits will be saved to the event as well as clusters
    # this is useful for redoing the clustering later
    produce_RecHits = cms.untracked.bool(True),
    # cell threshold in preshower
    thresh_PS = cms.double(7e-06),
    # n neighbours in ECAL 
    nNeighbours_Ecal = cms.int32(8),
    caloTowers = cms.InputTag("towerMakerPF"),
    hcalRecHitsHBHE = cms.InputTag("hbhereco"),
    # p1 for position calculation in HCAL 
    posCalcP1_Hcal = cms.double(1.0),
    ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    # all thresholds are in GeV
    # cell threshold in ECAL barrel 
    thresh_Ecal_Barrel = cms.double(0.08),
    # n neighbours in HCAL 
    nNeighbours_Hcal = cms.int32(4),
    # cell threshold in HCAL endcap 
    thresh_Hcal_Endcap = cms.double(0.8),
    ecalRecHitsES = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    # perform clustering in HCAL ?
    process_Hcal = cms.untracked.bool(True),
    #double thresh_Hcal_Barrel = 0.
    # seed threshold in HCAL barrel 
    thresh_Seed_Hcal_Barrel = cms.double(1.4),
    # n crystals for position calculation in HCAL
    posCalcNCrystal_Hcal = cms.int32(5),
    # cell threshold in ECAL endcap 
    thresh_Ecal_Endcap = cms.double(0.3),
    # clustering in preshower ?
    clustering_PS = cms.untracked.bool(True),
    # n neighbours in PS 
    nNeighbours_PS = cms.int32(8),
    # depth correction for ECAL clusters:
    #   0: no depth correction
    #   1: electrons/photons - depth correction is proportionnal to E
    #   2: hadrons - depth correction is fixed
    depthCor_Mode = cms.int32(1),
    depthCor_B_preshower = cms.double(4.0),
    # seed threshold in ECAL barrel 
    thresh_Seed_Ecal_Barrel = cms.double(0.23),
    # p1 for position calculation in ECAL 
    posCalcP1_Ecal = cms.double(-1.0),
    # clustering in HCAL: use calotowers for navigation ?
    clustering_Hcal_CaloTowers = cms.untracked.bool(True),
    # n crystals for position calculation in ECAL
    posCalcNCrystal_Ecal = cms.int32(9),
    # seed threshold in ECAL endcap 
    thresh_Seed_Ecal_Endcap = cms.double(0.8),
    # seed threshold in preshower
    thresh_Seed_PS = cms.double(0.0005),
    # sigma of the shower in preshower     
    showerSigma_PS = cms.double(0.2),
    # perform clustering in ECAL ?
    process_Ecal = cms.untracked.bool(True),
    # sigma of the shower in ECAL 
    showerSigma_Ecal = cms.double(5.0),
    # clustering in HCAL ?
    clustering_Hcal = cms.untracked.bool(True),
    # n crystals for position calculation in PS
    posCalcNCrystal_PS = cms.int32(-1),
    #double thresh_Hcal_Endcap = 0.
    # seed threshold in HCAL endcap 
    thresh_Seed_Hcal_Endcap = cms.double(1.4),
    # sigma of the shower in HCAL     
    showerSigma_Hcal = cms.double(10.0),
    # under the preshower, the depth is smaller, but the material is 
    # the same
    depthCor_A_preshower = cms.double(0.89)
)


