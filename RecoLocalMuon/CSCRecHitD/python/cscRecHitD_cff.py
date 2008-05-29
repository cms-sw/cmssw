import FWCore.ParameterSet.Config as cms

#
#    Parameters for coordinate and uncertainty calculations
#    Do not change them freely...
#
# These are empirical values. 
# To be extracted from measurements (DB)
#
cscRecHitDParameters = cms.PSet(
    XTasymmetry_ME1b = cms.untracked.double(0.0),
    #
    #    a XT asymmetry model parameter
    XTasymmetry_ME1a = cms.untracked.double(0.0),
    XTasymmetry_ME41 = cms.untracked.double(0.0),
    ConstSyst_ME1b = cms.untracked.double(0.007),
    XTasymmetry_ME22 = cms.untracked.double(0.0),
    XTasymmetry_ME21 = cms.untracked.double(0.0),
    ConstSyst_ME21 = cms.untracked.double(0.0),
    ConstSyst_ME22 = cms.untracked.double(0.0),
    XTasymmetry_ME31 = cms.untracked.double(0.0),
    #
    #    constant systematics (in cm)
    ConstSyst_ME1a = cms.untracked.double(0.022),
    NoiseLevel_ME13 = cms.untracked.double(8.0),
    NoiseLevel_ME12 = cms.untracked.double(9.0),
    NoiseLevel_ME32 = cms.untracked.double(9.0),
    NoiseLevel_ME31 = cms.untracked.double(9.0),
    ConstSyst_ME31 = cms.untracked.double(0.0),
    ConstSyst_ME41 = cms.untracked.double(0.0),
    XTasymmetry_ME13 = cms.untracked.double(0.0),
    XTasymmetry_ME12 = cms.untracked.double(0.0),
    ConstSyst_ME12 = cms.untracked.double(0.0),
    ConstSyst_ME13 = cms.untracked.double(0.0),
    ConstSyst_ME32 = cms.untracked.double(0.0),
    XTasymmetry_ME32 = cms.untracked.double(0.0),
    #    3 time bins noise (in ADC counts)
    NoiseLevel_ME1a = cms.untracked.double(7.0),
    NoiseLevel_ME1b = cms.untracked.double(8.0),
    NoiseLevel_ME21 = cms.untracked.double(9.0),
    NoiseLevel_ME22 = cms.untracked.double(9.0),
    NoiseLevel_ME41 = cms.untracked.double(9.0)
)

