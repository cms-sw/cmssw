import FWCore.ParameterSet.Config as cms

#electron mva ids
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_PHYS14_PU20bx25_nonTrig_V1_cff as ele_phys14_nt

#photon mva ids
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_PHYS14_PU20bx25_nonTrig_V1_cff as pho_phys14_nt
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V0_cff as pho_spring15_nt

ele_mva_prod_name = 'electronMVAValueMapProducer'
pho_mva_prod_name = 'photonMVAValueMapProducer'

def setup_mva(val_pset,cat_pset,prod_name,mva_name):
    value_name = '%s:%sValues'%(prod_name,mva_name)
    cat_name   = '%s:%sCategories'%(prod_name,mva_name)
    setattr( val_pset, '%sValues'%mva_name, cms.InputTag(value_name) )
    setattr( cat_pset, '%sCategories'%mva_name, cms.InputTag(cat_name) )

egamma_modifications = cms.VPSet(
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromFloatValueMaps'),
              electron_config = cms.PSet( ),
              photon_config   = cms.PSet( phoFull5x5SigmaIEtaIPhi = cms.InputTag('photonIDValueMapProducer:phoFull5x5SigmaIEtaIPhi'),
                                          phoFull5x5E1x3          = cms.InputTag('photonIDValueMapProducer:phoFull5x5E1x3'),
                                          phoFull5x5E2x2          = cms.InputTag('photonIDValueMapProducer:phoFull5x5E2x2'),
                                          phoFull5x5E2x5Max       = cms.InputTag('photonIDValueMapProducer:phoFull5x5E2x5Max'),
                                          phoESEffSigmaRR         = cms.InputTag('photonIDValueMapProducer:phoESEffSigmaRR'),
                                          phoChargedIsolation     = cms.InputTag('photonIDValueMapProducer:phoChargedIsolation'),
                                          phoNeutralHadronIsolation = cms.InputTag('photonIDValueMapProducer:phoNeutralHadronIsolation'),
                                          phoPhotonIsolation      = cms.InputTag('photonIDValueMapProducer:phoPhotonIsolation'),
                                          phoWorstChargedIsolation = cms.InputTag('photonIDValueMapProducer:phoWorstChargedIsolation')
                                          )
              ),
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromIntValueMaps'),
              electron_config = cms.PSet( ),
              photon_config   = cms.PSet( )
              )
)

#setup the mva value maps to embed
setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_phys14_nt.mvaPhys14NonTrigClassName)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_phys14_nt.mvaPhys14NonTrigClassName)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_spring15_nt.mvaSpring15NonTrigClassName)
<<<<<<< HEAD
=======


#############################################################
# REGRESSION MODIFIERS
#############################################################

egamma_modifications.append(
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  
              autoDetectBunchSpacing = cms.bool(True),
              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              vertexCollection = cms.InputTag("offlinePrimaryVertices"),
              bunchSpacingTag = cms.InputTag("addPileupInfo:bunchSpacing"),
              electron_config = cms.PSet( ecalRefinedRegressionWeightFile = cms.string("RecoEgamma/EgammaTools/data/GBRLikelihood_Clustering_746_bx25_Electrons_GedGsfElectron_results.root"),
                                          sigmaIetaIphi = cms.InputTag('electronRegressionValueMapProducer:eleFull5x5SigmaIEtaIPhi'),
                                          e2x5Max       = cms.InputTag('electronRegressionValueMapProducer:e2x5Max'),
                                          e2x5Left      = cms.InputTag('electronRegressionValueMapProducer:e2x5Left'),
                                          e2x5Right     = cms.InputTag('electronRegressionValueMapProducer:e2x5Right'),
                                          e2x5Top       = cms.InputTag('electronRegressionValueMapProducer:e2x5Top'),
                                          e2x5Bottom    = cms.InputTag('electronRegressionValueMapProducer:e2x5Bottom'),
                                          eMax          = cms.InputTag("electronRegressionValueMapProducer:eMax"),
                                          e2nd          = cms.InputTag("electronRegressionValueMapProducer:e2nd"),
                                          eTop          = cms.InputTag("electronRegressionValueMapProducer:eTop"),
                                          eBottom       = cms.InputTag("electronRegressionValueMapProducer:eBottom"),
                                          eLeft         = cms.InputTag("electronRegressionValueMapProducer:eLeft"),
                                          eRight        = cms.InputTag("electronRegressionValueMapProducer:eRight"),
                                          e3x3          = cms.InputTag("electronRegressionValueMapProducer:e3x3"),
                                          iPhi          = cms.InputTag("electronRegressionValueMapProducer:iPhi"),
                                          iEta          = cms.InputTag("electronRegressionValueMapProducer:iEta"),
                                          cryPhi        = cms.InputTag("electronRegressionValueMapProducer:cryPhi"),
                                          cryEta        = cms.InputTag("electronRegressionValueMapProducer:cryEta"),
                                          conditionsMean50ns  = cms.vstring("EBCorrection","EECorrection"),
                                          conditionsSigma50ns = cms.vstring("EBUncertainty","EEUncertainty"),
                                          conditionsMean25ns  = cms.vstring("EBCorrection","EECorrection"),
                                          conditionsSigma25ns = cms.vstring("EBUncertainty","EEUncertainty"),
                                          ),
              photon_config   = cms.PSet( photonRegressionWeightFile = cms.string("RecoEgamma/EgammaTools/data/regweights_forest_v2015_25ns_globalposition_ph.root"),
                                          sigmaIetaIphi = cms.InputTag('photonRegressionValueMapProducer:sigmaIetaIphi'),
                                          sigmaIphiIphi = cms.InputTag('photonRegressionValueMapProducer:sigmaIphiIphi'),
                                          e2x5Max       = cms.InputTag('photonRegressionValueMapProducer:e2x5Max'),
                                          e2x5Left      = cms.InputTag('photonRegressionValueMapProducer:e2x5Left'),
                                          e2x5Right     = cms.InputTag('photonRegressionValueMapProducer:e2x5Right'),
                                          e2x5Top       = cms.InputTag('photonRegressionValueMapProducer:e2x5Top'),
                                          e2x5Bottom    = cms.InputTag('photonRegressionValueMapProducer:e2x5Bottom'),
                                          conditionsMean50ns  = cms.vstring("EGRegressionForest_EB","EGRegressionForest_EE"),
                                          conditionsSigma50ns = cms.vstring("EGRegressionErrForest_EB","EGRegressionErrForest_EE"),
                                          conditionsMean25ns  = cms.vstring("EGRegressionForest_EB","EGRegressionForest_EE"),
                                          conditionsSigma25ns = cms.vstring("EGRegressionErrForest_EB","EGRegressionErrForest_EE"),
                                          )
              )
)
>>>>>>> 604e1cd... first commit of regression modifier
