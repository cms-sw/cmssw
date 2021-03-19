import FWCore.ParameterSet.Config as cms

ct2ct = cms.EDProducer("CaloTowersReCreator",
    # Weighting factor for EB   
    EBWeight = cms.double(1.0),
    HBGrid = cms.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    # energy scale for each subdetector (only Eb-Ee-Hb-He interpolations are available for now)
    HBEScale = cms.double(50.0),
    EEWeights = cms.vdouble(0.51, 1.39, 1.71, 2.37, 2.32, 
        2.2, 2.1, 1.98, 1.8),
    HF2Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEGrid = cms.vdouble(2.0, 4.0, 5.0, 9.0, 20.0, 
        30.0, 50.0, 100.0, 300.0),
    # Weighting factor for HE 10-degree cells   
    HEDWeight = cms.double(1.0),
    # Weighting factor for EE   
    EEWeight = cms.double(1.0),
    HBWeights = cms.vdouble(2.0, 1.86, 1.69, 1.55, 1.37, 
        1.19, 1.13, 1.11, 1.09, 1.0),
    # Weighting factor for HF long-fiber readouts 
    HF1Weight = cms.double(1.0),
    HF2Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeights = cms.vdouble(1.7, 1.57, 1.54, 1.49, 1.41, 
        1.26, 1.19, 1.15, 1.12, 1.0),
    HF1Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeights = cms.vdouble(0.86, 1.47, 1.66, 2.01, 1.98, 
        1.86, 1.83, 1.74, 1.65),
    # Weighting factor for HO 
    HOWeight = cms.double(1.0),
    # Weighting factor for HE 5-degree cells   
    HESWeight = cms.double(1.0),
    # Weighting factor for HF short-fiber readouts
    HF2Weight = cms.double(1.0),
    # Label of input CaloTowerCollection to use
    caloLabel = cms.InputTag("calotowermaker"),
    HF1Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HESEScale = cms.double(50.0),
    HESGrid = cms.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    HEDEScale = cms.double(50.0),
    HESWeights = cms.vdouble(1.7, 1.57, 1.54, 1.49, 1.41, 
        1.26, 1.19, 1.15, 1.12, 1.0),
    HEDGrid = cms.vdouble(0.0, 2.0, 4.0, 5.0, 9.0, 
        20.0, 30.0, 50.0, 100.0, 1000.0),
    EBEScale = cms.double(50.0),
    # Weighting factor for HB   
    HBWeight = cms.double(1.0),
    EEEScale = cms.double(50.0),
    HOGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    # Energy dependent weights and energy scale to be used
    EBGrid = cms.vdouble(2.0, 4.0, 5.0, 9.0, 20.0, 
        30.0, 50.0, 100.0, 300.0),
    # momentum assignment
    # Method for momentum reconstruction
    MomConstrMethod = cms.int32(1),                           
    # Depth, fraction of the respective calorimeter [0,1]
    MomHBDepth = cms.double(0.2),
    MomHEDepth = cms.double(0.4),   
    MomEBDepth = cms.double(0.3),
    MomEEDepth = cms.double(0.0),
	HcalPhase = cms.int32(0)
)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(ct2ct, HcalPhase = 1)

