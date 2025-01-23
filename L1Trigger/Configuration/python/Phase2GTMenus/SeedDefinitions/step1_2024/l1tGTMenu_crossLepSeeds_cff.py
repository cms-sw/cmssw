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

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_constants import *
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenuObjects_cff import *

TkMuonTkIsoEle720 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        regionsMinPt=cms.vdouble(7,7,7), # no scaling used below 8 GeV
    ),
    collection2 = l1tGTtkIsoElectron.clone(
        regionsMinPt = get_object_thrs(20, "CL2Electrons","Iso"),
    ),
    maxDz = cms.double(1),
)
pTkMuonTkIsoEle7_20 = cms.Path(TkMuonTkIsoEle720)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonTkIsoEle7_20")))

TkMuonTkEle723 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        regionsMinPt=cms.vdouble(7,7,7), # no scaling used below 8 GeV
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(23, "CL2Electrons","NoIso"),
    ),
    maxDz = cms.double(1),
)
pTkMuonTkEle7_23 = cms.Path(TkMuonTkEle723)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonTkEle7_23")))

TkEleTkMuon1020 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(36, "CL2Electrons","NoIso"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(20, "GMTTkMuons","VLoose"),
    ),
    maxDz = cms.double(1),
)
pTkEleTkMuon10_20 = cms.Path(TkEleTkMuon1020)
algorithms.append(cms.PSet(expression = cms.string("pTkEleTkMuon10_20")))

TkMuonDoubleTkEle61717 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        regionsMinPt=cms.vdouble(6,6,6),
    ),
    collection2 = l1tGTtkElectronLowPt.clone(
        regionsMinPt = get_object_thrs(17, "CL2Electrons","NoIso"),
    ),
    collection3 = l1tGTtkElectronLowPt.clone(
        regionsMinPt = get_object_thrs(17, "CL2Electrons","NoIso"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1)
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1)
    ),
)
pTkMuonDoubleTkEle6_17_17 = cms.Path(TkMuonDoubleTkEle61717)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonDoubleTkEle6_17_17")))

DoubleTkMuonTkEle559 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        regionsMinPt=cms.vdouble(5,5,5),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        regionsMinPt=cms.vdouble(5,5,5),
    ),
    collection3 = l1tGTtkElectronLowPt.clone(
        regionsMinPt = get_object_thrs(9, "CL2Electrons","NoIso"),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1)
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1)
    ),
)
pDoubleTkMuonTkEle5_5_9 = cms.Path(DoubleTkMuonTkEle559)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuonTkEle5_5_9")))

PuppiTauTkMuon4218 = l1tGTDoubleObjectCond.clone(  ###NB We need puppivertex here
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(18, "GMTTkMuons","VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0) # primary vertex index (choose 0) 
    ),
    collection2 = l1tGTnnTau.clone(
        regionsMinPt = get_object_thrs(42, "CL2Taus","default"),
    ),
)
pPuppiTauTkMuon42_18 = cms.Path(PuppiTauTkMuon4218)
algorithms.append(cms.PSet(expression = cms.string("pPuppiTauTkMuon42_18")))

PuppiTauTkIsoEle4522 = l1tGTDoubleObjectCond.clone(  ###NB We need puppivertex here
    collection1 = l1tGTtkIsoElectron.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(22, "CL2Electrons","Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0) # primary vertex index (choose 0) 
    ),
    collection2 = l1tGTnnTau.clone(
        regionsMinPt = get_object_thrs(45, "CL2Taus","default"),
    ),
)
pPuppiTauTkIsoEle45_22 = cms.Path(PuppiTauTkIsoEle4522)
algorithms.append(cms.PSet(expression = cms.string("pPuppiTauTkIsoEle45_22")))


