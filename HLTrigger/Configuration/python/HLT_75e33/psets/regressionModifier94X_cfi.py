import FWCore.ParameterSet.Config as cms

regressionModifier94X = cms.PSet(
    eOverP_ECALTRKThr = cms.double(0.025),
    electron_config = cms.PSet(
        regressionKey = cms.vstring(
            'electron_eb_ECALonly_lowpt',
            'electron_eb_ECALonly',
            'electron_ee_ECALonly_lowpt',
            'electron_ee_ECALonly',
            'electron_eb_ECALTRK_lowpt',
            'electron_eb_ECALTRK',
            'electron_ee_ECALTRK_lowpt',
            'electron_ee_ECALTRK'
        ),
        uncertaintyKey = cms.vstring(
            'electron_eb_ECALonly_lowpt_var',
            'electron_eb_ECALonly_var',
            'electron_ee_ECALonly_lowpt_var',
            'electron_ee_ECALonly_var',
            'electron_eb_ECALTRK_lowpt_var',
            'electron_eb_ECALTRK_var',
            'electron_ee_ECALTRK_lowpt_var',
            'electron_ee_ECALTRK_var'
        )
    ),
    epDiffSig_ECALTRKThr = cms.double(15.0),
    epSig_ECALTRKThr = cms.double(10.0),
    forceHighEnergyEcalTrainingIfSaturated = cms.bool(True),
    highEnergy_ECALTRKThr = cms.double(200.0),
    lowEnergy_ECALTRKThr = cms.double(50.0),
    lowEnergy_ECALonlyThr = cms.double(99999.0),
    modifierName = cms.string('EGRegressionModifierV2'),
    photon_config = cms.PSet(
        regressionKey = cms.vstring(
            'photon_eb_ECALonly_lowpt',
            'photon_eb_ECALonly',
            'photon_ee_ECALonly_lowpt',
            'photon_ee_ECALonly'
        ),
        uncertaintyKey = cms.vstring(
            'photon_eb_ECALonly_lowpt_var',
            'photon_eb_ECALonly_var',
            'photon_ee_ECALonly_lowpt_var',
            'photon_ee_ECALonly_var'
        )
    ),
    rhoCollection = cms.InputTag("fixedGridRhoFastjetAllTmp")
)