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
