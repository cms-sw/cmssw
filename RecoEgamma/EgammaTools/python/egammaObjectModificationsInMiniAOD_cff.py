import FWCore.ParameterSet.Config as cms

#electron mva ids
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff as ele_spring15_nt

#photon mva ids
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V0_cff as pho_spring15_25_nt
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2_cff as pho_spring15_50_nt

ele_mva_prod_name = 'electronMVAValueMapProducer'
pho_mva_prod_name = 'photonMVAValueMapProducer'

def setup_mva(val_pset,cat_pset,prod_name,mva_name):
    value_name = '%s:%sValues'%(prod_name,mva_name)
    cat_name   = '%s:%sCategories'%(prod_name,mva_name)
    setattr( val_pset, '%sValues'%mva_name, cms.InputTag(value_name) )
    setattr( cat_pset, '%sCategories'%mva_name, cms.InputTag(cat_name) )

egamma_modifications = cms.VPSet(
    cms.PSet( modifierName  = cms.string('EGFull5x5ShowerShapeModifierFromValueMaps'),
              photon_config = cms.PSet( sigmaIetaIeta = cms.InputTag('photonIDValueMapProducer:phoFull5x5SigmaIEtaIEta'),
                                        e5x5          = cms.InputTag('photonIDValueMapProducer:phoFull5x5E5x5')
                                        ) 
              ),
    cms.PSet( modifierName    = cms.string('EGPfIsolationModifierFromValueMaps'),
              photon_config   = cms.PSet( chargedHadronIso         = cms.InputTag('photonIDValueMapProducer:phoChargedIsolation'),
                                          neutralHadronIsolation   = cms.InputTag('photonIDValueMapProducer:phoNeutralHadronIsolation'),
                                          photonIso                = cms.InputTag('photonIDValueMapProducer:phoPhotonIsolation'),
                                          chargedHadronIsoWrongVtx = cms.InputTag('photonIDValueMapProducer:phoWorstChargedIsolation')
                                          )
              ),
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromFloatValueMaps'),
              electron_config = cms.PSet( ),
              photon_config   = cms.PSet( phoFull5x5SigmaIEtaIPhi = cms.InputTag('photonIDValueMapProducer:phoFull5x5SigmaIEtaIPhi'),
                                          phoFull5x5E1x3          = cms.InputTag('photonIDValueMapProducer:phoFull5x5E1x3'),
                                          phoFull5x5E2x2          = cms.InputTag('photonIDValueMapProducer:phoFull5x5E2x2'),
                                          phoFull5x5E2x5Max       = cms.InputTag('photonIDValueMapProducer:phoFull5x5E2x5Max'),
                                          phoESEffSigmaRR         = cms.InputTag('photonIDValueMapProducer:phoESEffSigmaRR'),
                                          )
              ),
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromIntValueMaps'),
              electron_config = cms.PSet( ),
              photon_config   = cms.PSet( )
              )
)

#setup the mva value maps to embed
setup_mva(egamma_modifications[2].electron_config,
          egamma_modifications[3].electron_config,
          ele_mva_prod_name,
          ele_spring15_nt.mvaSpring15NonTrigClassName+ele_spring15_nt.mvaTag)

setup_mva(egamma_modifications[2].photon_config,
          egamma_modifications[3].photon_config,
          pho_mva_prod_name,
          pho_spring15_25_nt.mvaSpring15NonTrigClassName+pho_spring15_25_nt.mvaTag)

setup_mva(egamma_modifications[2].photon_config,
          egamma_modifications[3].photon_config,
          pho_mva_prod_name,
          pho_spring15_50_nt.mvaSpring15NonTrigClassName+pho_spring15_50_nt.mvaTag)

#############################################################
# REGRESSION MODIFIERS
#############################################################

egamma_modifications.append(
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  
              autoDetectBunchSpacing = cms.bool(True),
              bunchSpacingTag = cms.InputTag("addPileupInfo:bunchSpacing"),
              manualBunchSpacing = cms.int32(50),              
              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              vertexCollection = cms.InputTag("offlinePrimaryVertices"),
              electron_config = cms.PSet( sigmaIetaIphi = cms.InputTag('electronRegressionValueMapProducer:eleFull5x5SigmaIEtaIPhi'),
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

              photon_config   = cms.PSet( sigmaIetaIphi = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5SigmaIEtaIPhi'),
                                          sigmaIphiIphi = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5SigmaIPhiIPhi'),
                                          e2x5Max       = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5E2x5Max'),
                                          e2x5Left      = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5E2x5Left'),
                                          e2x5Right     = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5E2x5Right'),
                                          e2x5Top       = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5E2x5Top'),
                                          e2x5Bottom    = cms.InputTag('photonRegressionValueMapProducer:phoFull5x5E2x5Bottom'),

                                          # EB, EE
                                          regressionKey_25ns  = cms.vstring('gedphoton_EBCorrection_25ns', 'gedphoton_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedphoton_EBUncertainty_25ns', 'gedphoton_EEUncertainty_25ns'),
                                          
                                          regressionKey_50ns  = cms.vstring('gedphoton_EBCorrection_50ns', 'gedphoton_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedphoton_EBUncertainty_50ns', 'gedphoton_EEUncertainty_50ns'),
                                          )
              )
)
