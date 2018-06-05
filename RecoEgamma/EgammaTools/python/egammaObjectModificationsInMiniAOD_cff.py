import FWCore.ParameterSet.Config as cms

#electron mva ids
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff as ele_spring15_nt
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_50ns_Trig_V1_cff as ele_spring15_50_t

import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff as ele_spring15_25_t

import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff as ele_spring16_gp_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff as ele_spring16_hzz_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff as ele_fall17_iso_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff as ele_fall17_noIso_v1

#photon mva ids
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff as pho_spring15_25_nt
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2p1_cff as pho_spring15_50_nt
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
          ele_spring15_nt.mvaSpring15NonTrigClassName+ele_spring15_nt.mvaTag)

setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_spring15_50_t.mvaSpring15TrigClassName+ele_spring15_50_t.mvaTag)

setup_mva(egamma_modifications[0].electron_config,
          egamma_modifications[1].electron_config,
          ele_mva_prod_name,
          ele_spring15_25_t.mvaSpring15TrigClassName+ele_spring15_25_t.mvaTag)

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
          pho_spring15_25_nt.mvaSpring15NonTrigClassName+pho_spring15_25_nt.mvaTag)

setup_mva(egamma_modifications[0].photon_config,
          egamma_modifications[1].photon_config,
          pho_mva_prod_name,
          pho_spring15_50_nt.mvaSpring15NonTrigClassName+pho_spring15_50_nt.mvaTag)

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
import RecoEgamma.EgammaTools.calibratedElectronProducerTRecoGsfElectron_cfi
for valueMapName in RecoEgamma.EgammaTools.calibratedElectronProducerTRecoGsfElectron_cfi.calibratedElectronProducerTRecoGsfElectron.valueMapsStored:
    setattr(reducedEgammaEnergyScaleAndSmearingModifier.electron_config,valueMapName,cms.InputTag("reducedEgamma",prefixName("calibEle",valueMapName)))

import RecoEgamma.EgammaTools.calibratedPhotonProducerTRecoPhoton_cfi
for valueMapName in RecoEgamma.EgammaTools.calibratedPhotonProducerTRecoPhoton_cfi.calibratedPhotonProducerTRecoPhoton.valueMapsStored:
    setattr(reducedEgammaEnergyScaleAndSmearingModifier.photon_config,valueMapName,cms.InputTag("reducedEgamma",prefixName("calibPho",valueMapName)))

#############################################################
# 8X to 9X modifiers (fills in variables new to 9X w.r.t 8X)
#############################################################
egamma8XObjectUpdateModifier = cms.PSet(
    modifierName  = cms.string('EG8XObjectUpdateModifier'),
    ecalRecHitsEB = cms.InputTag("reducedEgamma","reducedEBRecHits"),
    ecalRecHitsEE = cms.InputTag("reducedEgamma","reducedEERecHits"),
)

#############################################################
# 8X legacy needs an extra Et scale systematic
# due to an inflection around 45 GeV which is handled as a
# patch on top of the standard scale and smearing systematics
#############################################################
from RecoEgamma.EgammaTools.calibratedEgammas_cff import ecalTrkCombinationRegression
egamma8XLegacyEtScaleSysModifier = cms.PSet(
    modifierName = cms.string('EGEtScaleSysModifier'),
    epCombConfig = ecalTrkCombinationRegression,
    uncertFunc = cms.PSet(
        name = cms.string("UncertFuncV1"),
        lowEt = cms.double(43.5),
        highEt = cms.double(46.5),
        lowEtUncert = cms.double(0.002),
        highEtUncert = cms.double(-0.002)
        )
    )

def appendReducedEgammaEnergyScaleAndSmearingModifier(modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)

def prependEgamma8XObjectUpdateModifier(modifiers):
    modifiers.insert(0,egamma8XObjectUpdateModifier)

def appendEgamma8XLegacyAppendableModifiers (modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)
    modifiers.append(egamma8XLegacyEtScaleSysModifier)

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
run2_miniAOD_94XFall17.toModify(egamma_modifications,appendReducedEgammaEnergyScaleAndSmearingModifier)
   
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(egamma_modifications,appendEgamma8XLegacyAppendableModifiers)
run2_miniAOD_80XLegacy.toModify(egamma_modifications,prependEgamma8XObjectUpdateModifier)
