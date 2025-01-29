import FWCore.ParameterSet.Config as cms

############################################################
# Common objects for P2GT L1 seeds
############################################################

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_constants import *

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
)
l1tGTtkMuonVLoose = l1tGTtkMuonLoose.clone(
    qualityFlags = get_object_ids("GMTTkMuons","VLoose"),
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
)

l1tGTsc4Jet_er5 = l1tGTsc4Jet.clone(
    minEta = cms.double(-5),
    maxEta = cms.double(5),
)

############################################################
# Taus
############################################################
l1tGTnnTau = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
    minEta = cms.double(-2.172),
    maxEta = cms.double(2.172),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("CL2Taus"),
    minQualityScore = get_object_ids("CL2Taus","default")
)

############################################################
# Sums
############################################################

l1tGTHtSum = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum")
)

l1tGTEtSum = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "CL2EtSum")
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
)

l1tGTtkElectronLowPt = l1tGTtkElectronBase.clone(
    regionsQualityFlags = get_object_ids("CL2Electrons","NoIsoLowPt"),
)

l1tGTtkIsoElectron = l1tGTtkElectronBase.clone(
    regionsMaxRelIsolationPt = get_object_isos("CL2Electrons","Iso"),
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
)

l1tGTtkIsoPhoton = l1tGTtkPhoton.clone(
    regionsMaxRelIsolationPt = get_object_isos("CL2Photons","Iso"),
)