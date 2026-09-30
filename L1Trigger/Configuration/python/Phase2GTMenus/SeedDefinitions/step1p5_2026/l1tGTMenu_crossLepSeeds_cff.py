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

####### CROSS LEPTON SEEDS ###########

TkMu7TkEle20 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(9, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(24, "CL2Electrons", "NoIso"),
    ),
    maxDz = cms.double(1),
)
pTkMu7_TkEle20 = cms.Path(TkMu7TkEle20)
algorithms.append(cms.PSet(expression = cms.string("pTkMu7_TkEle20")))

TkMu20TkEle10 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(23, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(13, "CL2Electrons", "NoIso"),
    ),
    maxDz = cms.double(1),
)
pTkMu20_TkEle10 = cms.Path(TkMu20TkEle10)
algorithms.append(cms.PSet(expression = cms.string("pTkMu20_TkEle10")))

TkMuDoubleTkEle = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(8, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(15, "CL2Electrons", "NoIso"),
    ),
    collection3 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(15, "CL2Electrons", "NoIso"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1),
    ),
)
pTkMu_DoubleTkEle = cms.Path(TkMuDoubleTkEle)
algorithms.append(cms.PSet(expression = cms.string("pTkMu_DoubleTkEle")))

DoubleTkMuTkEle = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(7, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(7, "GMTTkMuons", "VLoose"),
    ),
    collection3 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(12, "CL2Electrons", "NoIso"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1),
    ),
)
pDoubleTkMu_TkEle = cms.Path(DoubleTkMuTkEle)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu_TkEle")))

DoubleTkMuDoubleTkEle = l1tGTQuadObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.3),
        maxEta = cms.double(2.3),
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.3),
        maxEta = cms.double(2.3),
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
    ),
    collection3 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(9, "CL2Electrons", "NoIso"),
    ),
    collection4 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(9, "CL2Electrons", "NoIso"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1),
        maxInvMass = cms.double(14),
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl14 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl34 = cms.PSet(
        maxInvMass = cms.double(20),
    ),
)
pDoubleTkMu_DoubleTkEle = cms.Path(DoubleTkMuDoubleTkEle)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu_DoubleTkEle")))
