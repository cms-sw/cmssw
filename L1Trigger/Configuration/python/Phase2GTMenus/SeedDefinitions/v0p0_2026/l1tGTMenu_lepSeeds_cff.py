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

# Step 1.5 menu, translated from Phase2-L1MenuTools configs/V50nano/rate_table/step1p5_menu_cfg.yml
# Path names follow the MenuTools seed names (L1_X -> pX) for direct rate comparisons.

####### MUON SEEDS ###########

SingleTkMuBarrel = l1tGTSingleObjectCond.clone(
    l1tGTtkMuonMedium.clone(
        minEta = cms.double(-0.83),
        maxEta = cms.double(0.83),
    ),
    regionsMinPt = get_object_thrs(13, "GMTTkMuons", "Medium"),
)
pSingleTkMu_Barrel = cms.Path(SingleTkMuBarrel)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkMu_Barrel")))

SingleTkMu = l1tGTSingleObjectCond.clone(
    l1tGTtkMuonVLoose.clone(),
    regionsMinPt = get_object_thrs(25, "GMTTkMuons", "VLoose"),
)
pSingleTkMu = cms.Path(SingleTkMu)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkMu")))

DoubleTkMu0er1p4SQOSdRMax1p4 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-1.4),
        maxEta = cms.double(1.4),
        regionsMinPt = get_object_thrs(1, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-1.4),
        maxEta = cms.double(1.4),
        regionsMinPt = get_object_thrs(1, "GMTTkMuons", "VLoose"),
    ),
    maxDR = cms.double(1.4),
    maxDz = cms.double(1),
    os = cms.bool(True),
)
pDoubleTkMu0er1p4_SQ_OS_dR_Max1p4 = cms.Path(DoubleTkMu0er1p4SQOSdRMax1p4)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu0er1p4_SQ_OS_dR_Max1p4")))

DoubleTkMu4er2p0SQOSdRMax1p6 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        regionsMinPt = get_object_thrs(6, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        regionsMinPt = get_object_thrs(6, "GMTTkMuons", "VLoose"),
    ),
    maxDR = cms.double(1.6),
    maxDz = cms.double(1),
    os = cms.bool(True),
)
pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6 = cms.Path(DoubleTkMu4er2p0SQOSdRMax1p6)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6")))

DoubleTkMu = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(10, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(10, "GMTTkMuons", "VLoose"),
    ),
    maxDz = cms.double(1),
)
pDoubleTkMu = cms.Path(DoubleTkMu)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMu")))

TripleTkMu = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
    ),
    collection3 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(5, "GMTTkMuons", "VLoose"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl23 = cms.PSet(),
)
pTripleTkMu = cms.Path(TripleTkMu)
algorithms.append(cms.PSet(expression = cms.string("pTripleTkMu")))

# MenuTools requires |sum of charges| == 1, i.e. at least one opposite-sign pair.
# P2GT has no charge-sum cut, so this is the OR of three conditions with os on each pair.
_TripleTkMu42p50MassMax12Base = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(6, "GMTTkMuons", "VLoose"),
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(4, "GMTTkMuons", "VLoose"),
    ),
    collection3 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(1, "GMTTkMuons", "VLoose"),
    ),
    correl12 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl13 = cms.PSet(
        maxDz = cms.double(1),
    ),
    correl23 = cms.PSet(),
    maxInvMass = cms.double(12),  # three-body mass
)
TripleTkMu42p50OS12MassMax12 = _TripleTkMu42p50MassMax12Base.clone(
    correl12 = cms.PSet(maxDz = cms.double(1), os = cms.bool(True)),
)
TripleTkMu42p50OS13MassMax12 = _TripleTkMu42p50MassMax12Base.clone(
    correl13 = cms.PSet(maxDz = cms.double(1), os = cms.bool(True)),
)
TripleTkMu42p50OS23MassMax12 = _TripleTkMu42p50MassMax12Base.clone(
    correl23 = cms.PSet(os = cms.bool(True)),
)
pTripleTkMu_4_2p5_0_OS12_Mass_Max12 = cms.Path(TripleTkMu42p50OS12MassMax12)
pTripleTkMu_4_2p5_0_OS13_Mass_Max12 = cms.Path(TripleTkMu42p50OS13MassMax12)
pTripleTkMu_4_2p5_0_OS23_Mass_Max12 = cms.Path(TripleTkMu42p50OS23MassMax12)
algorithms.append(cms.PSet(
    name = cms.string("pTripleTkMu_4_2p5_0_OS_Mass_Max12"),
    expression = cms.string("pTripleTkMu_4_2p5_0_OS12_Mass_Max12 or pTripleTkMu_4_2p5_0_OS13_Mass_Max12 or pTripleTkMu_4_2p5_0_OS23_Mass_Max12"),
))

####### EG and PHO SEEDS ###########

SingleTkEle = l1tGTSingleObjectCond.clone(
    l1tGTtkElectron.clone(),
    regionsMinPt = get_object_thrs(42, "CL2Electrons", "NoIso"),
)
pSingleTkEle = cms.Path(SingleTkEle)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkEle")))

SingleTkIsoPhoton = l1tGTSingleObjectCond.clone(
    l1tGTtkIsoPhoton.clone(),
    regionsMinPt = get_object_thrs(42, "CL2Photons", "Iso"),
)
pSingleTkIsoPhoton = cms.Path(SingleTkIsoPhoton)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkIsoPhoton")))

SingleTkEleIso = l1tGTSingleObjectCond.clone(
    l1tGTtkIsoElectron.clone(),
    regionsMinPt = get_object_thrs(35, "CL2Electrons", "Iso"),
)
pSingleTkEleIso = cms.Path(SingleTkEleIso)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkEleIso")))

DoubleTkEleer1p2dRMax0p6 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkElectron.clone(
        minEta = cms.double(-1.2),
        maxEta = cms.double(1.2),
        regionsMinPt = get_object_thrs(18, "CL2Electrons", "NoIso"),
    ),
    collection2 = l1tGTtkElectron.clone(
        minEta = cms.double(-1.2),
        maxEta = cms.double(1.2),
        regionsMinPt = get_object_thrs(14, "CL2Electrons", "NoIso"),
    ),
    maxDR = cms.double(0.6),
    maxDz = cms.double(1),
)
pDoubleTkEle_er1p2_dR_Max0p6 = cms.Path(DoubleTkEleer1p2dRMax0p6)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle_er1p2_dR_Max0p6")))

DoubleTkEleer1p5dEtaMax1p5 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkElectron.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
        regionsMinPt = get_object_thrs(18, "CL2Electrons", "NoIso"),
    ),
    collection2 = l1tGTtkElectron.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
        regionsMinPt = get_object_thrs(18, "CL2Electrons", "NoIso"),
    ),
    maxDEta = cms.double(1.5),
    maxDz = cms.double(1),
)
pDoubleTkEle_er1p5_dEta_Max1p5 = cms.Path(DoubleTkEleer1p5dEtaMax1p5)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle_er1p5_dEta_Max1p5")))

DoubleTkEle = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(30, "CL2Electrons", "NoIso"),
    ),
    collection2 = l1tGTtkElectron.clone(
        regionsMinPt = get_object_thrs(15, "CL2Electrons", "NoIso"),
    ),
    maxDz = cms.double(1),
)
pDoubleTkEle = cms.Path(DoubleTkEle)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle")))

DoubleTkPhoIso = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoPhoton.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
        regionsMinPt = get_object_thrs(24, "CL2Photons", "Iso"),
    ),
    collection2 = l1tGTtkIsoPhoton.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
        regionsMinPt = get_object_thrs(15, "CL2Photons", "Iso"),
    ),
)
pDoubleTkPhoIso = cms.Path(DoubleTkPhoIso)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkPhoIso")))

# Standalone EG: as in Step 1, CL2 photons without isolation with the L1EG scalings
SingleEGEle = l1tGTSingleObjectCond.clone(
    l1tGTtkPhoton.clone(),
    regionsMinPt = get_object_thrs(51, "L1EG", "default"),
)
pSingleEGEle = cms.Path(SingleEGEle)
algorithms.append(cms.PSet(expression = cms.string("pSingleEGEle")))

DoubleEGEle = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkPhoton.clone(
        regionsMinPt = get_object_thrs(37, "L1EG", "default"),
    ),
    collection2 = l1tGTtkPhoton.clone(
        regionsMinPt = get_object_thrs(24, "L1EG", "default"),
    ),
)
pDoubleEGEle = cms.Path(DoubleEGEle)
algorithms.append(cms.PSet(expression = cms.string("pDoubleEGEle")))

####### TAU SEEDS ###########

# L1_SingleTau (CaloTau >= 153, |eta| < 2.1) and L1_DoubleTau (CaloTau 92,92, |eta| < 2.1)
# are not implemented: there is no calo tau collection in the P2GT emulator (as for Step 1).

DoubleIsoTau = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(58, "CL2Taus", "default"),
    ),
    collection2 = l1tGTnnTau.clone(
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt = get_object_thrs(58, "CL2Taus", "default"),
    ),
)
pDoubleIsoTau = cms.Path(DoubleIsoTau)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTau")))
