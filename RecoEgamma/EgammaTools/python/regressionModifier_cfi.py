import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGRegressionModifierV2'),  

              rhoCollection = cms.InputTag('fixedGridRhoFastjetAll'),
              
              electron_config = cms.PSet( # EB, EE
                regressionKey_ecalonly  = cms.vstring('electron_eb_ECALonly_lowpt', 'electron_eb_ECALonly', 'electron_ee_ECALonly_lowpt', 'electron_ee_ECALonly'),
                uncertaintyKey_ecalonly = cms.vstring('electron_eb_ECALonly_lowpt_var', 'electron_eb_ECALonly_var', 'electron_ee_ECALonly_lowpt_var', 'electron_ee_ECALonly_var'),
                regressionKey_ecaltrk  = cms.vstring('electron_eb_ECALTRK_lowpt', 'electron_eb_ECALTRK', 'electron_ee_ECALTRK_lowpt', 'electron_ee_ECALTRK'),
                uncertaintyKey_ecaltrk = cms.vstring('electron_eb_ECALTRK_lowpt_var', 'electron_eb_ECALTRK_var', 'electron_ee_ECALTRK_lowpt_var', 'electron_ee_ECALTRK_var'),
                                          ),
              
              photon_config   = cms.PSet( # EB, EE
                regressionKey_ecalonly  = cms.vstring('photon_eb_ECALonly_lowpt', 'photon_eb_ECALonly', 'photon_ee_ECALonly_lowpt', 'photon_ee_ECALonly'),
                uncertaintyKey_ecalonly = cms.vstring('photon_eb_ECALonly_lowpt_var', 'photon_eb_ECALonly_var', 'photon_ee_ECALonly_lowpt_var', 'photon_ee_ECALonly_var'),
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
