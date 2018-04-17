import FWCore.ParameterSet.Config as cms

#electron mva ids
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff as ele_spring16_gp_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff as ele_spring16_hzz_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff as ele_fall17_iso_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff as ele_fall17_noIso_v1


#photon mva ids
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff as pho_spring16_nt_v1
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff as pho_fall17_94X_v1
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff as pho_fall17_94X_v1p1


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
              photon_config   = cms.PSet( phoChargedIsolation         = cms.InputTag('photonIDValueMapProducer:phoChargedIsolation'),
                                          phoNeutralHadronIsolation   = cms.InputTag('photonIDValueMapProducer:phoNeutralHadronIsolation'),
                                          phoPhotonIsolation          = cms.InputTag('photonIDValueMapProducer:phoPhotonIsolation'),
                                          phoWorstChargedIsolation    = cms.InputTag('photonIDValueMapProducer:phoWorstChargedIsolation')
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
          ele_spring16_gp_v1.mvaSpring16ClassName+ele_spring16_gp_v1.mvaTag)

setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_spring16_hzz_v1.mvaSpring16ClassName+ele_spring16_hzz_v1.mvaTag)

setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_fall17_iso_v1.mvaFall17ClassName+ele_fall17_iso_v1.mvaTag)

setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_fall17_noIso_v1.mvaFall17ClassName+ele_fall17_noIso_v1.mvaTag)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_spring16_nt_v1.mvaSpring16NonTrigClassName+pho_spring16_nt_v1.mvaTag)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_fall17_94X_v1.mvaFall17v1ClassName+pho_fall17_94X_v1.mvaTag)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_fall17_94X_v1p1.mvaFall17v1p1ClassName+pho_fall17_94X_v1p1.mvaTag)

#############################################################
# REGRESSION MODIFIERS
#############################################################

#from RecoEgamma.EgammaTools.regressionModifier_cfi import *

#egamma_modifications.append( regressionModifier )

#############################################################
# Scale and Smearing Modifiers
#############################################################
reducedEgammaEnergyScaleAndSmearingModifier = cms.PSet(
    modifierName    = cms.string('EGExtraInfoModifierFromFloatValueMaps'),
    electron_config = cms.PSet(),
    photon_config   = cms.PSet()
)
from RecoEgamma.EgammaTools.calibratedEgammas_cff import prefixName
import RecoEgamma.EgammaTools.calibratedElectronProducer_cfi
for valueMapName in RecoEgamma.EgammaTools.calibratedElectronProducer_cfi.calibratedElectronProducer.valueMapsStored:
    setattr(reducedEgammaEnergyScaleAndSmearingModifier.electron_config,valueMapName,cms.InputTag("reducedEgamma",prefixName("calibEle",valueMapName)))

import RecoEgamma.EgammaTools.calibratedPhotonProducer_cfi
for valueMapName in RecoEgamma.EgammaTools.calibratedPhotonProducer_cfi.calibratedPhotonProducer.valueMapsStored:
    setattr(reducedEgammaEnergyScaleAndSmearingModifier.photon_config,valueMapName,cms.InputTag("reducedEgamma",prefixName("calibPho",valueMapName)))


def appendReducedEgammaEnergyScaleAndSmearingModifier(modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
run2_miniAOD_94XFall17.toModify(egamma_modifications,appendReducedEgammaEnergyScaleAndSmearingModifier)
   
