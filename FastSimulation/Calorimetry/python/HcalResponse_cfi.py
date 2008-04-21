import FWCore.ParameterSet.Config as cms

HCALResponseBlock = cms.PSet(
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
        # If needed - add a smal energy to each hadron ...
        energyBias = cms.double(0.0),
        ElectronForwardResolution_Noise = cms.double(0.0)
    )
)

