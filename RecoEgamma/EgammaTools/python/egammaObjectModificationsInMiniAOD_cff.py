import FWCore.ParameterSet.Config as cms

#electron mva ids
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff as ele_spring16_gp_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff as ele_spring16_hzz_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff as ele_fall17_iso_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff as ele_fall17_noIso_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff as ele_fall17_iso_v2
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff as ele_fall17_noIso_v2
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff as ele_summer18UL_hzz

import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff as ele_RunIIIWinter22_iso_v1
import RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff as ele_RunIIIWinter22_noIso_v1

#photon mva ids
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff as pho_spring16_nt_v1
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff as pho_fall17_94X_v1p1
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff as pho_fall17_94X_v2
import RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Winter22_122X_V1_cff as pho_winter22_122X_v1


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
              photon_config   = cms.PSet( )
              ),
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromIntValueMaps'),
              electron_config = cms.PSet( ),
              photon_config   = cms.PSet( )
              )
)

#setup the mva value maps to embed
for ele_mva_cff in [
          ele_spring16_gp_v1,
          ele_spring16_hzz_v1,
          ele_fall17_iso_v1,
          ele_fall17_noIso_v1,
          ele_fall17_iso_v2,
          ele_fall17_noIso_v2,
          ele_summer18UL_hzz,
          ele_RunIIIWinter22_iso_v1,
          ele_RunIIIWinter22_noIso_v1
        ]:

    setup_mva(egamma_modifications[0].electron_config,
              egamma_modifications[1].electron_config,
              ele_mva_prod_name,
              ele_mva_cff.mvaClassName + ele_mva_cff.mvaTag)

for pho_mva_cff in [
          pho_spring16_nt_v1,
          pho_fall17_94X_v1p1,
          pho_fall17_94X_v2,
          pho_winter22_122X_v1
        ]:

    setup_mva(egamma_modifications[0].photon_config,
              egamma_modifications[1].photon_config,
              pho_mva_prod_name,
              pho_mva_cff.mvaClassName + pho_mva_cff.mvaTag)

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

#############################################################
# 8X to 9X modifiers (fills in variables new to 9X w.r.t 8X)
#############################################################
egamma8XObjectUpdateModifier = cms.PSet(
    modifierName  = cms.string('EG8XObjectUpdateModifier'),
    ecalRecHitsEB = cms.InputTag("reducedEgamma","reducedEBRecHits"),
    ecalRecHitsEE = cms.InputTag("reducedEgamma","reducedEERecHits"),
)

#############################################################
# 9X-106X to 106X modifiers (fills in variables new to 106X w.r.t 9X-105X)
#############################################################
egamma9X105XUpdateModifier = cms.PSet( 
    modifierName    = cms.string('EG9X105XObjectUpdateModifier'),
    eleCollVMsAreKeyedTo = cms.InputTag("slimmedElectrons",processName=cms.InputTag.skipCurrentProcess()),
    phoCollVMsAreKeyedTo = cms.InputTag("slimmedPhotons",processName=cms.InputTag.skipCurrentProcess()),
    conversions = cms.InputTag("reducedEgamma","reducedConversions"),
    beamspot = cms.InputTag("offlineBeamSpot"),
    ecalRecHitsEB = cms.InputTag("reducedEgamma","reducedEBRecHits"),
    ecalRecHitsEE = cms.InputTag("reducedEgamma","reducedEERecHits"),
    eleTrkIso = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso"),
    eleTrkIso04 = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso04"),
    phoPhotonIso = cms.InputTag("photonIDValueMapProducer","phoPhotonIsolation"),
    phoNeutralHadIso = cms.InputTag("photonIDValueMapProducer","phoNeutralHadronIsolation"),
    phoChargedHadIso = cms.InputTag("photonIDValueMapProducer","phoChargedIsolation"),
    phoChargedHadWorstVtxIso = cms.InputTag("photonIDValueMapProducer","phoWorstChargedIsolation"),
    phoChargedHadWorstVtxConeVetoIso = cms.InputTag("photonIDValueMapProducer","phoWorstChargedIsolationConeVeto"),
    phoChargedHadPFPVIso = cms.InputTag("egmPhotonIsolation","h+-DR030-"),
    allowGsfTrackForConvs = cms.bool(False),
    updateChargedHadPFPVIso = cms.bool(True)
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

# modifier for photon isolation used in heavy ions
egammaHIPhotonIsolationModifier = cms.PSet(
    modifierName = cms.string('EGExtraInfoModifierFromHIPhotonIsolationValueMaps'),
    electron_config = cms.PSet(),
    photon_config = cms.PSet(
        photonIsolationHI = cms.InputTag("reducedEgamma:photonIsolationHIProducerppGED")
        )
    )

photonDRNModifier = cms.PSet(
      modifierName = cms.string("EGRegressionModifierDRN"),
      patPhotons = cms.PSet(
          source = cms.InputTag("selectedPatPhotons"),
          correctionsSource = cms.InputTag('patPhotonsDRN'),
          energyFloat = cms.string("energyDRN"),
          resFloat = cms.string("resolutionDRN")
        )
    )

def appendReducedEgammaEnergyScaleAndSmearingModifier(modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)

def prependEgamma8XObjectUpdateModifier(modifiers):
    modifiers.insert(0,egamma8XObjectUpdateModifier)

def appendEgamma8XLegacyAppendableModifiers (modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)
    modifiers.append(egamma8XLegacyEtScaleSysModifier)

def appendEgammaHIPhotonIsolationModifier(modifiers):
    modifiers.append(egammaHIPhotonIsolationModifier)

def appendPhotonDRNModifier(modifiers):
    modifiers.append(photonDRNModifier)

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
(run2_miniAOD_94XFall17 | run2_miniAOD_UL).toModify(egamma_modifications,appendReducedEgammaEnergyScaleAndSmearingModifier)
   
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
#80X doesnt have the bug which prevents GsfTracks used to match conversions so set true
run2_miniAOD_80XLegacy.toModify(egamma9X105XUpdateModifier,allowGsfTrackForConvs = True)
run2_miniAOD_80XLegacy.toModify(egamma_modifications,appendEgamma8XLegacyAppendableModifiers)
run2_miniAOD_80XLegacy.toModify(egamma_modifications,prependEgamma8XObjectUpdateModifier)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(egamma_modifications, appendEgammaHIPhotonIsolationModifier)

from Configuration.ProcessModifiers.photonDRN_cff import _photonDRN
_photonDRN.toModify(egamma_modifications, appendPhotonDRNModifier)
