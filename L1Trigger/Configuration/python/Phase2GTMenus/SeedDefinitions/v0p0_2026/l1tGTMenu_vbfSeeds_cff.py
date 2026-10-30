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

####### VBF SEEDS ###########

DoubleJetMassMin1000 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(145, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(70, "CL2JetsSC4", "default"),
    ),
    minInvMass = cms.double(1000),
)
pDoubleJet_Mass_Min1000 = cms.Path(DoubleJetMassMin1000)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_Mass_Min1000")))

DoubleJetIsoTau = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(61, "CL2Taus", "default"),
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
pDoubleJet_IsoTau = cms.Path(DoubleJetIsoTau)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_IsoTau")))

DoubleJetTkEleIso = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(93, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTtkIsoElectron.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(24, "CL2Electrons", "Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    correl12 = cms.PSet(
        minInvMass = cms.double(550),
    ),
)
pDoubleJet_TkEleIso = cms.Path(DoubleJetTkEleIso)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_TkEleIso")))

DoubleJetTkMu = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(133, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(83, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    correl12 = cms.PSet(
        minInvMass = cms.double(700),
    ),
)
pDoubleJet_TkMu = cms.Path(DoubleJetTkMu)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_TkMu")))

DoubleJetMETSum = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(118, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(83, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTEtSum.clone(
        minPt = get_object_thrs(176, "CL2EtSum", "default"),
    ),
    correl12 = cms.PSet(
        minInvMass = cms.double(600),
    ),
)
pDoubleJet_METSum = cms.Path(DoubleJetMETSum)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_METSum")))
