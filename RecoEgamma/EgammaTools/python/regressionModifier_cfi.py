import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  
              autoDetectBunchSpacing = cms.bool(True),
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
    
