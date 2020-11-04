import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff import interestingEgammaIsoHCALSel

reducedEgamma = cms.EDProducer("ReducedEGProducer",
  keepPhotons = cms.string("hadTowOverEm()<0.15 && pt>10 && (pt>14 || chargedHadronIso()<10)"), #keep in output
  slimRelinkPhotons = cms.string("hadTowOverEm()<0.15 && pt>10 && (pt>14 || chargedHadronIso()<10)"), #keep only slimmed SuperCluster plus seed cluster
  relinkPhotons = cms.string("(r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), #keep all associated clusters/rechits/conversions
  keepOOTPhotons = cms.string("pt>10"), #keep in output
  slimRelinkOOTPhotons = cms.string("pt>10"), #keep only slimmed SuperCluster plus seed cluster
  relinkOOTPhotons = cms.string("(r9()>0.8)"), #keep all associated clusters/rechits/conversions
  keepGsfElectrons = cms.string(""), #keep in output
  slimRelinkGsfElectrons = cms.string(""), #keep only slimmed SuperCluster plus seed cluster
  relinkGsfElectrons = cms.string("pt>5"), #keep all associated clusters/rechits/conversions
  photons = cms.InputTag("gedPhotons"),
  ootPhotons = cms.InputTag("ootPhotons"),
  gsfElectrons = cms.InputTag("gedGsfElectrons"),
  conversions = cms.InputTag("allConversions"),
  gsfTracks = cms.InputTag("electronGsfTracks"),
  singleConversions = cms.InputTag("particleFlowEGamma"),
  barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB"),
  endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE"),
  preshowerEcalHits = cms.InputTag("reducedEcalRecHitsES"),
  hbheHits= cms.InputTag("reducedHcalRecHits","hbhereco"),
  photonsPFValMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
  gsfElectronsPFValMap = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
  photonIDSources = cms.VInputTag(),
  photonIDOutput = cms.vstring(),
  gsfElectronIDSources = cms.VInputTag(),
  gsfElectronIDOutput = cms.vstring(),
  photonFloatValueMapSources = cms.VInputTag(),
  photonFloatValueMapOutput = cms.vstring(),
  ootPhotonFloatValueMapSources = cms.VInputTag(),
  ootPhotonFloatValueMapOutput = cms.vstring(),
  gsfElectronFloatValueMapSources = cms.VInputTag(),
  gsfElectronFloatValueMapOutput = cms.vstring(),
  applyPhotonCalibOnData = cms.bool(False),
  applyPhotonCalibOnMC = cms.bool(False),
  applyGsfElectronCalibOnData = cms.bool(False),
  applyGsfElectronCalibOnMC = cms.bool(False), 
  photonCalibEnergySource = cms.InputTag(""),
  photonCalibEnergyErrSource = cms.InputTag(""),
  gsfElectronCalibEnergySource = cms.InputTag(""),
  gsfElectronCalibEnergyErrSource = cms.InputTag(""),
  gsfElectronCalibEcalEnergySource = cms.InputTag(""),
  gsfElectronCalibEcalEnergyErrSource = cms.InputTag(""),
  hcalHitSel = interestingEgammaIsoHCALSel
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(reducedEgamma, 
        preshowerEcalHits = cms.InputTag(""),
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(
    reducedEgamma, 
    photonFloatValueMapSources = cms.VInputTag(
        cms.InputTag("photonEcalPFClusterIsolationProducer"),
        cms.InputTag("photonHcalPFClusterIsolationProducer"),
        ),
    photonFloatValueMapOutput = cms.vstring(
        "phoEcalPFClusIso",
        "phoHcalPFClusIso",
        ),
    ootPhotonFloatValueMapSources = cms.VInputTag(
        cms.InputTag("ootPhotonEcalPFClusterIsolationProducer"),
        ),
    ootPhotonFloatValueMapOutput = cms.vstring(
        "ootPhoEcalPFClusIso",
        ),
    gsfElectronFloatValueMapSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducer"),
        cms.InputTag("electronHcalPFClusterIsolationProducer"),
        ),
    gsfElectronFloatValueMapOutput = cms.vstring(
        "eleEcalPFClusIso",
        "eleHcalPFClusIso",
        )
    )

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
run2_miniAOD_94XFall17.toModify(
    reducedEgamma, 
    photonFloatValueMapSources = cms.VInputTag(
        cms.InputTag("photonEcalPFClusterIsolationProducer"),
        cms.InputTag("photonHcalPFClusterIsolationProducer"),
        ),
    photonFloatValueMapOutput = cms.vstring(
        "phoEcalPFClusIso",
        "phoHcalPFClusIso",
        ),
    ootPhotonFloatValueMapSources = cms.VInputTag(
        cms.InputTag("ootPhotonEcalPFClusterIsolationProducer"),
        cms.InputTag("ootPhotonHcalPFClusterIsolationProducer"),
        ),
    ootPhotonFloatValueMapOutput = cms.vstring(
        "ootPhoEcalPFClusIso",
        "ootPhoHcalPFClusIso",
        ),
    gsfElectronFloatValueMapSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducer"),
        cms.InputTag("electronHcalPFClusterIsolationProducer"),
        ),
    gsfElectronFloatValueMapOutput = cms.vstring(
        "eleEcalPFClusIso",
        "eleHcalPFClusIso",
        )
    )

from RecoEgamma.EgammaPhotonProducers.reducedEgamma_tools import calibrateReducedEgamma
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
modifyReducedEGammaRun2MiniAOD = (
    run2_miniAOD_94XFall17 | run2_miniAOD_80XLegacy | run2_miniAOD_UL).makeProcessModifier(calibrateReducedEgamma)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(reducedEgamma,
    ootPhotons = cms.InputTag(""),
    keepPhotons = "pt>15 && abs(eta)<2.5",
    slimRelinkPhotons = "pt>15 && abs(eta)<2.5",
    relinkPhotons = "pt>15 && abs(eta)<2.5"
    )
