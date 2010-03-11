import FWCore.ParameterSet.Config as cms

#
#    Parameters for coordinate and uncertainty calculations
#    Do not change them freely...
#
# These are empirical values. 
# To be extracted from measurements (DB)
# The values bellow were tuned on MC (though there were MTCC ones too)
#
cscRecHitDParameters = cms.PSet(
    #
    #    a XT asymmetry model parameter
    XTasymmetry_ME1a = cms.double(0.023),
    XTasymmetry_ME1b = cms.double(0.01),
    XTasymmetry_ME12 = cms.double(0.015),
    XTasymmetry_ME13 = cms.double(0.02),
    XTasymmetry_ME21 = cms.double(0.023),
    XTasymmetry_ME22 = cms.double(0.023),
    XTasymmetry_ME31 = cms.double(0.023),
    XTasymmetry_ME32 = cms.double(0.023),
    XTasymmetry_ME41 = cms.double(0.023),
    #
    #    constant systematics (in cm)
    ConstSyst_ME1a = cms.double(0.01),
    ConstSyst_ME1b = cms.double(0.02),
    ConstSyst_ME12 = cms.double(0.02),
    ConstSyst_ME13 = cms.double(0.03),
    ConstSyst_ME21 = cms.double(0.03),
    ConstSyst_ME22 = cms.double(0.03),
    ConstSyst_ME31 = cms.double(0.03),
    ConstSyst_ME32 = cms.double(0.03),
    ConstSyst_ME41 = cms.double(0.03),
    #
    #    3 time bins noise (in ADC counts)
    NoiseLevel_ME1a = cms.double(9.0),
    NoiseLevel_ME1b = cms.double(6.0),
    NoiseLevel_ME12 = cms.double(7.0),
    NoiseLevel_ME13 = cms.double(4.0),
    NoiseLevel_ME21 = cms.double(5.0),
    NoiseLevel_ME22 = cms.double(7.0),
    NoiseLevel_ME31 = cms.double(5.0),
    NoiseLevel_ME32 = cms.double(7.0),
    NoiseLevel_ME41 = cms.double(5.0)
)

