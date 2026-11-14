import FWCore.ParameterSet.Config as cms

############################################################
# Common objects for P2GT L1 seeds
#
# Objects and pT scalings are top-level PSets that the seeds
# reference by name (object = gt_ref("..."), see l1tGTObject_constants),
# so each definition exists once and stays editable (e.g. in ConfDB).
# 'ptScaling' is the default offline->online scaling used with
# offlineMinPt/offlineMinScalarSumPt; a seed can override it.
############################################################

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_constants import *

############################################################
# Offline pT scalings (from l1tGTObject_scalings)
############################################################

l1tGTScaling_GMTTkMuons_VLoose = get_object_scaling("GMTTkMuons", "VLoose")
l1tGTScaling_GMTTkMuons_Loose = get_object_scaling("GMTTkMuons", "Loose")
l1tGTScaling_CL2JetsSC4_default = get_object_scaling("CL2JetsSC4", "default")
l1tGTScaling_CL2Taus_default = get_object_scaling("CL2Taus", "default")
l1tGTScaling_CL2HtSum_HT = get_object_scaling("CL2HtSum", "HT")
l1tGTScaling_CL2HtSum_MHT = get_object_scaling("CL2HtSum", "MHT")
l1tGTScaling_CL2EtSum_default = get_object_scaling("CL2EtSum", "default")
l1tGTScaling_CL2Electrons_NoIso = get_object_scaling("CL2Electrons", "NoIso")
l1tGTScaling_CL2Electrons_Iso = get_object_scaling("CL2Electrons", "Iso")
l1tGTScaling_CL2Photons_Iso = get_object_scaling("CL2Photons", "Iso")
l1tGTScaling_L1EG_default = get_object_scaling("L1EG", "default")

############################################################
# Muons
############################################################

l1tGTtkMuon = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("GMTTkMuons"),
)
l1tGTtkMuonLoose = l1tGTtkMuon.clone(
    qualityFlags = get_object_ids("GMTTkMuons","Loose"),
    ptScaling = gt_ref("l1tGTScaling_GMTTkMuons_Loose"),
)
l1tGTtkMuonVLoose = l1tGTtkMuonLoose.clone(
    qualityFlags = get_object_ids("GMTTkMuons","VLoose"),
    ptScaling = gt_ref("l1tGTScaling_GMTTkMuons_VLoose"),
)

# GMT standalone displaced muons (no track match, so no z0 and no online-to-offline scaling)
l1tGTsaDisplacedMuon = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "GMTSaDisplacedMuons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
)

############################################################
# Jets
############################################################

l1tGTsc4Jet = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
    minEta = cms.double(-2.4),
    maxEta = cms.double( 2.4),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2JetsSC4"),
    # minPt = cms.double(25), # safety cut - can be enabled everywhere (for now done in the get_threshold function)
    ptScaling = gt_ref("l1tGTScaling_CL2JetsSC4_default"),
)

l1tGTsc4Jet_er5 = l1tGTsc4Jet.clone(
    minEta = cms.double(-5),
    maxEta = cms.double(5),
)

l1tGTsc8Jet_er5 = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2JetsSC8"),
    minEta = cms.double(-5),
    maxEta = cms.double(5),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2JetsSC8"),
)

############################################################
# Taus
############################################################
l1tGTnnTau = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
    minEta = cms.double(-2.172),
    maxEta = cms.double(2.172),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2Taus"),
    minQualityScore = get_object_ids("CL2Taus","default"),
    ptScaling = gt_ref("l1tGTScaling_CL2Taus_default"),
)

############################################################
# Sums
############################################################

l1tGTHtSum = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    ptScaling = gt_ref("l1tGTScaling_CL2HtSum_HT"), # MHT seeds use l1tGTScaling_CL2HtSum_MHT
)

l1tGTEtSum = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),
    ptScaling = gt_ref("l1tGTScaling_CL2EtSum_default"),
)

############################################################
# Electrons
############################################################

l1tGTtkElectronBase = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2Electrons"),
)

l1tGTtkElectron = l1tGTtkElectronBase.clone(
    regionsQualityFlags = get_object_ids("CL2Electrons","NoIso"),
    ptScaling = gt_ref("l1tGTScaling_CL2Electrons_NoIso"),
)

l1tGTtkElectronLowPt = l1tGTtkElectronBase.clone(
    regionsQualityFlags = get_object_ids("CL2Electrons","NoIsoLowPt"),
    ptScaling = gt_ref("l1tGTScaling_CL2Electrons_NoIso"),
)

l1tGTtkIsoElectron = l1tGTtkElectronBase.clone(
    regionsMaxRelIsolationPt = get_object_isos("CL2Electrons","Iso"),
    regionsQualityFlags = get_object_ids("CL2Electrons","Iso"),
    ptScaling = gt_ref("l1tGTScaling_CL2Electrons_Iso"),
)

############################################################
# Photons
############################################################

l1tGTtkPhoton = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2Photons"),
    regionsQualityFlags = get_object_ids("CL2Photons","Iso"),
    ptScaling = gt_ref("l1tGTScaling_L1EG_default"), # used for the EG seeds
)

l1tGTtkIsoPhoton = l1tGTtkPhoton.clone(
    regionsMaxRelIsolationPt = get_object_isos("CL2Photons","Iso"),
    ptScaling = gt_ref("l1tGTScaling_CL2Photons_Iso"),
)