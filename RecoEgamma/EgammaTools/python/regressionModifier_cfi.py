import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  
              autoDetectBunchSpacing = cms.bool(True),
              bunchSpacingTag = cms.InputTag("addPileupInfo:bunchSpacing"),
              manualBunchSpacing = cms.int32(50),              
              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              vertexCollection = cms.InputTag("offlinePrimaryVertices"),
              electron_config = cms.PSet( sigmaIetaIphi = cms.InputTag('electronRegressionValueMapProducer:sigmaIEtaIPhi'),
                                          eMax          = cms.InputTag("electronRegressionValueMapProducer:eMax"),
                                          e2nd          = cms.InputTag("electronRegressionValueMapProducer:e2nd"),
                                          eTop          = cms.InputTag("electronRegressionValueMapProducer:eTop"),
                                          eBottom       = cms.InputTag("electronRegressionValueMapProducer:eBottom"),
                                          eLeft         = cms.InputTag("electronRegressionValueMapProducer:eLeft"),
                                          eRight        = cms.InputTag("electronRegressionValueMapProducer:eRight"),
                                          clusterMaxDR          = cms.InputTag("electronRegressionValueMapProducer:clusterMaxDR"),
                                          clusterMaxDRDPhi      = cms.InputTag("electronRegressionValueMapProducer:clusterMaxDRDPhi"),
                                          clusterMaxDRDEta      = cms.InputTag("electronRegressionValueMapProducer:clusterMaxDRDEta"),
                                          clusterMaxDRRawEnergy = cms.InputTag("electronRegressionValueMapProducer:clusterMaxDRRawEnergy"),
                                          clusterRawEnergy0     = cms.InputTag("electronRegressionValueMapProducer:clusterRawEnergy0"),
                                          clusterRawEnergy1     = cms.InputTag("electronRegressionValueMapProducer:clusterRawEnergy1"),
                                          clusterRawEnergy2     = cms.InputTag("electronRegressionValueMapProducer:clusterRawEnergy2"),
                                          clusterDPhiToSeed0    = cms.InputTag("electronRegressionValueMapProducer:clusterDPhiToSeed0"),
                                          clusterDPhiToSeed1    = cms.InputTag("electronRegressionValueMapProducer:clusterDPhiToSeed1"),
                                          clusterDPhiToSeed2    = cms.InputTag("electronRegressionValueMapProducer:clusterDPhiToSeed2"),
                                          clusterDEtaToSeed0    = cms.InputTag("electronRegressionValueMapProducer:clusterDEtaToSeed0"),
                                          clusterDEtaToSeed1    = cms.InputTag("electronRegressionValueMapProducer:clusterDEtaToSeed1"),
                                          clusterDEtaToSeed2    = cms.InputTag("electronRegressionValueMapProducer:clusterDEtaToSeed2"),
                                          iPhi          = cms.InputTag("electronRegressionValueMapProducer:iPhi"),
                                          iEta          = cms.InputTag("electronRegressionValueMapProducer:iEta"),
                                          cryPhi        = cms.InputTag("electronRegressionValueMapProducer:cryPhi"),
                                          cryEta        = cms.InputTag("electronRegressionValueMapProducer:cryEta"),
                                          intValueMaps = cms.vstring("iPhi", "iEta"),                                          
                                          
                                          # EB, EE
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
    
