import FWCore.ParameterSet.Config as cms

#Global fast calorimetry parameters
Calorimetry = cms.PSet(
    ECAL = cms.PSet(
        # For the core 10% of the spots for
        CoreIntervals = cms.vdouble(100.0, 0.1),
        # change the radius of the tail of the shower
        RTFactor = cms.double(1.0),
        # change the radius of the core of the shower
        RCFactor = cms.double(1.0),
        # For the tail 10% of r<1RM. 100% otherwise
        TailIntervals = cms.vdouble(1.0, 0.1, 100.0, 1.0),
        FrontLeakageProbability = cms.double(1.0),
        GridSize = cms.int32(7),
        # change globally the Moliere radius 
        RadiusFactor = cms.double(1.096),
        Debug = cms.untracked.bool(False),
        #SpotFraction < 0 <=> deactivated. In the case, CoreIntervals and 
        #TailIntervals are used   
        SpotFraction = cms.double(-1.0),
        GapLossProbability = cms.double(0.9)
    ),
    CalorimeterProperties = cms.PSet(
        PreshowerLayer2_thickness = cms.double(0.38),
        ECALEndcap_LightCollection = cms.double(0.023),
        PreshowerLayer1_thickness = cms.double(1.6),
        PreshowerLayer1_mipsPerGeV = cms.double(35.7),
        PreshowerLayer2_mipsPerGeV = cms.double(59.5),
        ECALBarrel_LightCollection = cms.double(0.03),
        HCAL_Sampling = cms.double(0.0035),
        # Watch out ! The following two values are defined wrt the electron shower simulation
        # There are not directly related to the detector properties
        HCAL_PiOverE = cms.double(0.2)
    ),
    UnfoldedMode = cms.untracked.bool(False),
    HCAL = cms.PSet(
        SimMethod = cms.int32(0), ## 0 - use HDShower, 1 - use HDRShower

        GridSize = cms.int32(7),
        #-- 0 - simple response, 1 - parametrized response + showering, 2 - tabulated response + showering
        SimOption = cms.int32(2)
    ),
    HSParameters = cms.PSet(
        nTRsteps = cms.int32(40),
        lossesOpt = cms.int32(0),
        depthStep = cms.double(0.5),
        balanceEH = cms.double(0.9),
        eSpotSize = cms.double(0.2),
        hcalDepthFactor = cms.double(1.1),
        transRparam = cms.double(1.0),
        nDepthSteps = cms.int32(10),
        maxTRfactor = cms.double(4.0),
        criticalHDEnergy = cms.double(3.0)
    ),
    HCALResponse = cms.PSet(
        eResponseCoefficient = cms.double(1.0),
        HadronEndcapResolution_Noise = cms.double(0.0),
        HadronForwardResolution_Stochastic = cms.double(1.82),
        ElectronForwardResolution_Constant = cms.double(0.05),
        HadronBarrelResolution_Noise = cms.double(0.0),
        HadronForwardResolution_Constant = cms.double(0.09),
        HadronBarrelResolution_Stochastic = cms.double(1.22),
        HadronEndcapResolution_Constant = cms.double(0.05),
        eResponseExponent = cms.double(1.0),
        HadronForwardResolution_Noise = cms.double(0.0),
        HadronBarrelResolution_Constant = cms.double(0.05),
        HadronEndcapResolution_Stochastic = cms.double(1.3),
        eResponseCorrection = cms.double(1.0),
        eResponseScaleHB = cms.double(3.0),
        eResponseScaleHF = cms.double(3.0),
        eResponseScaleHE = cms.double(3.0),
        ElectronForwardResolution_Stochastic = cms.double(1.38),
        eResponsePlateauHE = cms.double(0.95),
        eResponsePlateauHF = cms.double(0.95),
        eResponsePlateauHB = cms.double(0.95),
        energyBias = cms.double(0.0),
        ElectronForwardResolution_Noise = cms.double(0.0)
    )
)

