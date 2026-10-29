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

# In MenuTools these seeds include an L1PV leg with |z0(lepton) - z0(PV)| < 1 cm,
# implemented here as maxPrimVertDz on each lepton leg w.r.t. the leading primary vertex.

####### CROSS LEPTON-HADRONIC SEEDS ###########

TkMu6HTSum240 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(8, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(329, "CL2HtSum", "HT"),
    ),
)
pTkMu6_HTSum240 = cms.Path(TkMu6HTSum240)
algorithms.append(cms.PSet(expression = cms.string("pTkMu6_HTSum240")))

TkMu12HTSum150 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(15, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(219, "CL2HtSum", "HT"),
    ),
)
pTkMu12_HTSum150 = cms.Path(TkMu12HTSum150)
algorithms.append(cms.PSet(expression = cms.string("pTkMu12_HTSum150")))

DoubleTkMuHTSum = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(305, "CL2HtSum", "HT"),
    ),
)
pDoubleTkMu_HTSum = cms.Path(DoubleTkMuHTSum)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu_HTSum")))

TkMuIsoTau = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(21, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(55, "CL2Taus", "default"),
    ),
)
pTkMu_IsoTau = cms.Path(TkMuIsoTau)
algorithms.append(cms.PSet(expression = cms.string("pTkMu_IsoTau")))

TkMuJetMETSum = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(148, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTEtSum.clone(
        minPt = get_object_thrs(143, "CL2EtSum", "default"),
    ),
)
pTkMu_Jet_METSum = cms.Path(TkMuJetMETSum)
algorithms.append(cms.PSet(expression = cms.string("pTkMu_Jet_METSum")))

DoubleTkMuMETSumHTSum = l1tGTQuadObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = l1tGTEtSum.clone(
        minPt = get_object_thrs(143, "CL2EtSum", "default"),
    ),
    collection4 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(109, "CL2HtSum", "HT"),
    ),
)
pDoubleTkMu_METSum_HTSum = cms.Path(DoubleTkMuMETSumHTSum)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu_METSum_HTSum")))

# The jet must be within dR < 0.8 of either muon: OR of two conditions
DoubleTkMuJetMu1 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(1, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(1, "GMTTkMuons", "VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = l1tGTsc4Jet.clone(
        minEta = cms.double(-2.5),
        maxEta = cms.double(2.5),
        regionsMinPt = get_object_thrs(138, "CL2JetsSC4", "default"),
    ),
    correl12 = cms.PSet(
        maxDR = cms.double(1.6),
    ),
    correl13 = cms.PSet(
        maxDR = cms.double(0.8),
    ),
)
DoubleTkMuJetMu2 = DoubleTkMuJetMu1.clone(
    correl13 = cms.PSet(),
    correl23 = cms.PSet(
        maxDR = cms.double(0.8),
    ),
)
pDoubleTkMu_Jet_Mu1 = cms.Path(DoubleTkMuJetMu1)
pDoubleTkMu_Jet_Mu2 = cms.Path(DoubleTkMuJetMu2)
algorithms.append(cms.PSet(
    name = cms.string("pDoubleTkMu_Jet"),
    expression = cms.string("pDoubleTkMu_Jet_Mu1 or pDoubleTkMu_Jet_Mu2"),
))

DoubleTkEleHTSum = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(11, "CL2Electrons", "NoIso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(11, "CL2Electrons", "NoIso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(402, "CL2HtSum", "HT"),
    ),
)
pDoubleTkEle_HTSum = cms.Path(DoubleTkEleHTSum)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle_HTSum")))

TkEleIsoIsoTau = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoElectron.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(26, "CL2Electrons", "Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(56, "CL2Taus", "default"),
    ),
    minDR = cms.double(0.3),
)
pTkEleIso_IsoTau = cms.Path(TkEleIsoIsoTau)
algorithms.append(cms.PSet(expression = cms.string("pTkEleIso_IsoTau")))

TkEleIsoHTSum = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoElectron.clone(
        regionsMinPt = get_object_thrs(17, "CL2Electrons", "Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = l1tGTHtSum.clone(
        minScalarSumPt = get_object_thrs(280, "CL2HtSum", "HT"),
    ),
)
pTkEleIso_HTSum = cms.Path(TkEleIsoHTSum)
algorithms.append(cms.PSet(expression = cms.string("pTkEleIso_HTSum")))

####### CROSS HADRONIC SEEDS ###########

DoubleIsoTauJet = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(58, "CL2Taus", "default"),
    ),
    collection2 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(56, "CL2Taus", "default"),
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
pDoubleIsoTau_Jet = cms.Path(DoubleIsoTauJet)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTau_Jet")))
