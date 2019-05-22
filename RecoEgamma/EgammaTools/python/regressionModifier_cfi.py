import FWCore.ParameterSet.Config as cms

regressionModifier = cms.PSet(
    modifierName = cms.string('EGRegressionModifierV3'),       
    rhoTag = cms.InputTag('fixedGridRhoFastjetAllTmp'),
    useClosestToCentreSeedCrysDef = cms.bool(True),
    eleRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            rangeMin = cms.double(-1.),
            rangeMax = cms.double(3.0),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.string("electron_eb_ECALonly_lowpt"),
            ebHighEtForestName = cms.string("electron_eb_ECALonly"),
            eeLowEtForestName = cms.string("electron_ee_ECALonly_lowpt"),
            eeHighEtForestName = cms.string("electron_ee_ECALonly"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMin = cms.double(0.0002),
            rangeMax = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.string("electron_eb_ECALonly_lowpt_var"),
            ebHighEtForestName = cms.string("electron_eb_ECALonly_var"),
            eeLowEtForestName = cms.string("electron_ee_ECALonly_lowpt_var"),
            eeHighEtForestName = cms.string("electron_ee_ECALonly_var"),
            ),
        epComb = cms.PSet(
            ecalTrkRegressionConfig = cms.PSet(
                rangeMin = cms.double(-1.),
                rangeMax = cms.double(3.0),
                lowEtHighEtBoundary = cms.double(50.),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.string('electron_eb_ECALTRK_lowpt'),
                ebHighEtForestName = cms.string('electron_eb_ECALTRK'),
                eeLowEtForestName = cms.string('electron_ee_ECALTRK_lowpt'),
                eeHighEtForestName = cms.string('electron_ee_ECALTRK')
                ),
            ecalTrkRegressionUncertConfig = cms.PSet(
                rangeMin = cms.double(0.0002),
                rangeMax = cms.double(0.5),
                lowEtHighEtBoundary = cms.double(50.),  
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.string('electron_eb_ECALTRK_lowpt_var'),
                ebHighEtForestName = cms.string('electron_eb_ECALTRK_var'),
                eeLowEtForestName = cms.string('electron_ee_ECALTRK_lowpt_var'),
                eeHighEtForestName = cms.string('electron_ee_ECALTRK_var')
                ),
            maxEcalEnergyForComb=cms.double(200.),
            minEOverPForComb=cms.double(0.025),
            maxEPDiffInSigmaForComb=cms.double(15.),
            maxRelTrkMomErrForComb=cms.double(10.),                
            )
        )
)

regressionModifier94X = \
    cms.PSet( modifierName    = cms.string('EGRegressionModifierV2'),  

              rhoCollection = cms.InputTag('fixedGridRhoFastjetAll'),
              
              electron_config = cms.PSet( # EB, EE
                regressionKey  = cms.vstring('electron_eb_ECALonly_lowpt', 'electron_eb_ECALonly', 'electron_ee_ECALonly_lowpt', 'electron_ee_ECALonly',
                                             'electron_eb_ECALTRK_lowpt', 'electron_eb_ECALTRK', 'electron_ee_ECALTRK_lowpt', 'electron_ee_ECALTRK'),
                uncertaintyKey = cms.vstring('electron_eb_ECALonly_lowpt_var', 'electron_eb_ECALonly_var', 'electron_ee_ECALonly_lowpt_var', 'electron_ee_ECALonly_var',
                                             'electron_eb_ECALTRK_lowpt_var', 'electron_eb_ECALTRK_var', 'electron_ee_ECALTRK_lowpt_var', 'electron_ee_ECALTRK_var'),
                                          ),
              
              photon_config   = cms.PSet( # EB, EE
                regressionKey  = cms.vstring('photon_eb_ECALonly_lowpt', 'photon_eb_ECALonly', 'photon_ee_ECALonly_lowpt', 'photon_ee_ECALonly'),
                uncertaintyKey = cms.vstring('photon_eb_ECALonly_lowpt_var', 'photon_eb_ECALonly_var', 'photon_ee_ECALonly_lowpt_var', 'photon_ee_ECALonly_var'),
                                          ),

              lowEnergy_ECALonlyThr = cms.double(99999.),
              lowEnergy_ECALTRKThr = cms.double(50.),
              highEnergy_ECALTRKThr = cms.double(200.),
              eOverP_ECALTRKThr = cms.double(0.025),
              epDiffSig_ECALTRKThr = cms.double(15.),
              epSig_ECALTRKThr = cms.double(10.),
              forceHighEnergyEcalTrainingIfSaturated = cms.bool(True)

              )


regressionModifier80X = \
    cms.PSet( modifierName    = cms.string('EGRegressionModifierV1'),
              autoDetectBunchSpacing = cms.bool(True),
              applyExtraHighEnergyProtection = cms.bool(True),
              bunchSpacingTag = cms.InputTag("bunchSpacingProducer"),
              manualBunchSpacing = cms.int32(50),
              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              vertexCollection = cms.InputTag("offlinePrimaryVertices"),
              electron_config = cms.PSet( # EB, EE                                                                                                                                                          
                                          regressionKey_25ns  = cms.vstring('gedelectron_EBCorrection_25ns', 'gedelectron_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedelectron_EBUncertainty_25ns', 'gedelectron_EEUncertainty_25ns'),
                                          combinationKey_25ns   = cms.string('gedelectron_p4combination_25ns'),

                                          regressionKey_50ns  = cms.vstring('gedelectron_EBCorrection_50ns', 'gedelectron_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedelectron_EBUncertainty_50ns', 'gedelectron_EEUncertainty_50ns'),
                                          combinationKey_50ns   = cms.string('gedelectron_p4combination_50ns'),
                                          ),

              photon_config   = cms.PSet( # EB, EE                                                                                                                                                          
                                          regressionKey_25ns  = cms.vstring('gedphoton_EBCorrection_25ns', 'gedphoton_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedphoton_EBUncertainty_25ns', 'gedphoton_EEUncertainty_25ns'),

                                          regressionKey_50ns  = cms.vstring('gedphoton_EBCorrection_50ns', 'gedphoton_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedphoton_EBUncertainty_50ns', 'gedphoton_EEUncertainty_50ns'),
                                          )
              )
