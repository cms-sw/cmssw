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

DoubleTkMuonOSEr1p5Dr1p4 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minEta = cms.double(-1.5),
        maxEta = cms.double(1.5),
    ),
    minDR = cms.double(0),
    maxDR =cms.double(1.4),
    maxDz = cms.double(1),
    os = cms.bool(True),    
)
pDoubleTkMuon_OS_Er1p5_Dr1p4 = cms.Path(DoubleTkMuonOSEr1p5Dr1p4)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon_OS_Er1p5_Dr1p4")))

DoubleTkMuon44OSDr1p2 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(4),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(4),
    ),
    minDR = cms.double(0),
    maxDR = cms.double(1.2),
    maxDz = cms.double(1),
    os = cms.bool(True),    
)
pDoubleTkMuon_4_4_OS_Dr1p2 = cms.Path(DoubleTkMuon44OSDr1p2)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon_4_4_OS_Dr1p2")))

DoubleTkMuon4p5OSEr2Mass7to18 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        minPt = cms.double(4),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        minPt = cms.double(4),
    ),
    minDR = cms.double(0),
    minInvMass = cms.double(7),
    maxInvMass = cms.double(18),
    maxDz = cms.double(1),
    os = cms.bool(True),    
)
pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18 = cms.Path(DoubleTkMuon4p5OSEr2Mass7to18)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18")))

TripleTkMuon530OSMassMax9 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(5),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(3),
    ),
    collection3 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(0),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1),
        os = cms.bool(True),
        maxInvMass = cms.double(9),
    ),
    correl13 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1)
    ),
    correl23 = cms.PSet(
        minDR = cms.double(0),
    )
)
pTripleTkMuon_5_3_0_DoubleTkMuon_5_3_OS_MassTo9 = cms.Path(TripleTkMuon530OSMassMax9)
algorithms.append(cms.PSet(expression = cms.string("pTripleTkMuon_5_3_0_DoubleTkMuon_5_3_OS_MassTo9")))

TripleTkMuon53p52p5OSMass5to17 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(5),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(4),
    ),
    collection3 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(2),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1),
    ),
    correl13 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1),
        os = cms.bool(True),
        minInvMass = cms.double(5),
        maxInvMass = cms.double(17),
    ),
    correl23 = cms.PSet(
        minDR = cms.double(0),
    )
)
pTripleTkMuon_5_3p5_2p5_OS_Mass5to17 = cms.Path(TripleTkMuon53p52p5OSMass5to17)
algorithms.append(cms.PSet(expression = cms.string("pTripleTkMuon_5_3p5_2p5_OS_Mass5to17")))
