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

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1p5_2026.l1tGTObject_constants import *
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1p5_2026.l1tGTMenuObjects_cff import *

####### JET, MET, HT ###########

SingleJet = l1tGTSingleObjectCond.clone(
    l1tGTsc4Jet_er5.clone(),
    regionsMinPt = get_object_thrs(228, "CL2JetsSC4", "default"),
)
pSingleJet = cms.Path(SingleJet)
algorithms.append(cms.PSet(expression = cms.string("pSingleJet")))

DoubleJetMassMin250dEtaMax1p5 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(78, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(78, "CL2JetsSC4", "default"),
    ),
    minInvMass = cms.double(250),
    maxDEta = cms.double(1.5),
)
pDoubleJet_Mass_Min250_dEta_Max1p5 = cms.Path(DoubleJetMassMin250dEtaMax1p5)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_Mass_Min250_dEta_Max1p5")))

DoubleJetdEtaMax1p6 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.3),
        maxEta = cms.double(2.3),
        regionsMinPt = get_object_thrs(160, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.3),
        maxEta = cms.double(2.3),
        regionsMinPt = get_object_thrs(160, "CL2JetsSC4", "default"),
    ),
    maxDEta = cms.double(1.6),
)
pDoubleJet_dEta_Max1p6 = cms.Path(DoubleJetdEtaMax1p6)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet_dEta_Max1p6")))

DoubleJet = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(198, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(198, "CL2JetsSC4", "default"),
    ),
)
pDoubleJet = cms.Path(DoubleJet)
algorithms.append(cms.PSet(expression = cms.string("pDoubleJet")))

HTSum = l1tGTSingleObjectCond.clone(
    l1tGTHtSum.clone(),
    minScalarSumPt = get_object_thrs(378, "CL2HtSum", "HT"),
)
pHTSum = cms.Path(HTSum)
algorithms.append(cms.PSet(expression = cms.string("pHTSum")))

METSum = l1tGTSingleObjectCond.clone(
    l1tGTEtSum.clone(),
    minPt = get_object_thrs(210, "CL2EtSum", "default"),
)
pMETSum = cms.Path(METSum)
algorithms.append(cms.PSet(expression = cms.string("pMETSum")))
