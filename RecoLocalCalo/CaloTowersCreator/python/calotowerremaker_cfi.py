import FWCore.ParameterSet.Config as cms

ct2ct = cms.EDFilter("CaloTowersReCreator",
    # Weighting factor for EB   
    EBWeight = cms.double(1.0),
    HBGrid = cms.untracked.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    # energy scale for each subdetector (only Eb-Ee-Hb-He interpolations are available for now)
    HBEScale = cms.untracked.double(50.0),
    EEWeights = cms.untracked.vdouble(0.51, 1.39, 1.71, 2.37, 2.32, 
        2.2, 2.1, 1.98, 1.8),
    HF2Weights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEGrid = cms.untracked.vdouble(2.0, 4.0, 5.0, 9.0, 20.0, 
        30.0, 50.0, 100.0, 300.0),
    # Weighting factor for HE 10-degree cells   
    HEDWeight = cms.double(1.0),
    # Weighting factor for EE   
    EEWeight = cms.double(1.0),
    HBWeights = cms.untracked.vdouble(2.0, 1.86, 1.69, 1.55, 1.37, 
        1.19, 1.13, 1.11, 1.09, 1.0),
    # Weighting factor for HF long-fiber readouts 
    HF1Weight = cms.double(1.0),
    HF2Grid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeights = cms.untracked.vdouble(1.7, 1.57, 1.54, 1.49, 1.41, 
        1.26, 1.19, 1.15, 1.12, 1.0),
    HF1Grid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeights = cms.untracked.vdouble(0.86, 1.47, 1.66, 2.01, 1.98, 
        1.86, 1.83, 1.74, 1.65),
    # Weighting factor for HO 
    HOWeight = cms.double(1.0),
    # Weighting factor for HE 5-degree cells   
    HESWeight = cms.double(1.0),
    # Weighting factor for HF short-fiber readouts
    HF2Weight = cms.double(1.0),
    # Label of input CaloTowerCollection to use
    caloLabel = cms.InputTag("calotowermaker"),
    HF1Weights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HESEScale = cms.untracked.double(50.0),
    HESGrid = cms.untracked.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    HEDEScale = cms.untracked.double(50.0),
    HESWeights = cms.untracked.vdouble(1.7, 1.57, 1.54, 1.49, 1.41, 
        1.26, 1.19, 1.15, 1.12, 1.0),
    HEDGrid = cms.untracked.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    EBEScale = cms.untracked.double(50.0),
    # Weighting factor for HB   
    HBWeight = cms.double(1.0),
    EEEScale = cms.untracked.double(50.0),
    HOGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    # Energy dependent weights and energy scale to be used
    EBGrid = cms.untracked.vdouble(2.0, 4.0, 5.0, 9.0, 20.0, 
        30.0, 50.0, 100.0, 300.0),
    # CaloTower 4-momentum reconstruction method and parameters
    MomConstrMethod = cms.int32(0),
    MomEmDepth = cms.double(0),
    MomHadDepth = cms.double(0),
    MomTotDepth = cms.double(0)
)


