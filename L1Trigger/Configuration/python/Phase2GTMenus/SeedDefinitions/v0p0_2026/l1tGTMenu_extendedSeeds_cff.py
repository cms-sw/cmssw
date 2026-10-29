import FWCore.ParameterSet.Config as cms

############################################################
# L1 Global Trigger Emulation
############################################################

# Conditions

from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

from L1Trigger.Phase2L1GT.l1tGTSingleObjectCond_cfi import l1tGTSingleObjectCond
from L1Trigger.Phase2L1GT.l1tGTDoubleObjectCond_cfi import l1tGTDoubleObjectCond
from L1Trigger.Phase2L1GT.l1tGTTripleObjectCond_cfi import l1tGTTripleObjectCond
from L1Trigger.Phase2L1GT.l1tGTQuadObjectCond_cfi import l1tGTQuadObjectCond

from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.v0p0_2026.l1tGTObject_constants import *
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.v0p0_2026.l1tGTMenuObjects_cff import *

# Extended-eta seeds of the v0p0 menu, from Phase2-L1MenuTools configs/V50nano/rate_table/v0p0_menu_cfg.yml.
# Thresholds use the "extended" scalings, derived for the extended eta acceptance.
# L1_SingleTau_Ext and L1_DoubleTau_Ext are not implemented: there is no calo tau collection in P2GT.

####### PUPPI TAU SEEDS, |eta| < 2.4 ###########

DoubleIsoTauExt = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(58, "CL2Taus", "extended"),
    ),
    collection2 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(58, "CL2Taus", "extended"),
    ),
)
pDoubleIsoTau_Ext = cms.Path(DoubleIsoTauExt)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTau_Ext")))

TkMuIsoTauExt = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(21, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(55, "CL2Taus", "extended"),
    ),
)
pTkMu_IsoTau_Ext = cms.Path(TkMuIsoTauExt)
algorithms.append(cms.PSet(expression = cms.string("pTkMu_IsoTau_Ext")))

TkEleIsoIsoTauExt = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoElectron.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(26, "CL2Electrons", "Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(56, "CL2Taus", "extended"),
    ),
    minDR = cms.double(0.3),
)
pTkEleIso_IsoTau_Ext = cms.Path(TkEleIsoIsoTauExt)
algorithms.append(cms.PSet(expression = cms.string("pTkEleIso_IsoTau_Ext")))

DoubleIsoTauExtJet = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(58, "CL2Taus", "extended"),
    ),
    collection2 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(56, "CL2Taus", "extended"),
    ),
    collection3 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(103, "CL2JetsSC4", "default"),
    ),
    correl13 = cms.PSet(
        minDR = cms.double(0.5),
    ),
    correl23 = cms.PSet(
        minDR = cms.double(0.5),
    ),
)
pDoubleIsoTau_Ext_Jet = cms.Path(DoubleIsoTauExtJet)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTau_Ext_Jet")))

DoubleJetIsoTauExt = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTnnTauExt.clone(
        regionsMinPt = get_object_thrs(61, "CL2Taus", "extended"),
    ),
    correl12 = cms.PSet(
        minInvMass = cms.double(600),
    ),
    correl13 = cms.PSet(
        minDR = cms.double(0.5),
    ),
    correl23 = cms.PSet(
        minDR = cms.double(0.5),
    ),
)
pDoubleJet_IsoTau_Ext = cms.Path(DoubleJetIsoTauExt)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_IsoTau_Ext")))
