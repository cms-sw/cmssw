import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  
              autoDetectBunchSpacing = cms.bool(True),
              bunchSpacingTag = cms.InputTag("addPileupInfo:bunchSpacing"),
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
              
              photon_config   = cms.PSet( sigmaIetaIphi = cms.InputTag('photonRegressionValueMapProducer:sigmaIEtaIPhi'),
                                          sigmaIphiIphi = cms.InputTag('photonRegressionValueMapProducer:sigmaIPhiIPhi'),
                                          e2x5Max       = cms.InputTag('photonRegressionValueMapProducer:e2x5Max'),
                                          e2x5Left      = cms.InputTag('photonRegressionValueMapProducer:e2x5Left'),
                                          e2x5Right     = cms.InputTag('photonRegressionValueMapProducer:e2x5Right'),
                                          e2x5Top       = cms.InputTag('photonRegressionValueMapProducer:e2x5Top'),
                                          e2x5Bottom    = cms.InputTag('photonRegressionValueMapProducer:e2x5Bottom'),
                                          
                                          # EB, EE
                                          regressionKey_25ns  = cms.vstring('gedphoton_EBCorrection_25ns', 'gedphoton_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedphoton_EBUncertainty_25ns', 'gedphoton_EEUncertainty_25ns'),
                                          
                                          regressionKey_50ns  = cms.vstring('gedphoton_EBCorrection_50ns', 'gedphoton_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedphoton_EBUncertainty_50ns', 'gedphoton_EEUncertainty_50ns'),
                                          )
              )
    
