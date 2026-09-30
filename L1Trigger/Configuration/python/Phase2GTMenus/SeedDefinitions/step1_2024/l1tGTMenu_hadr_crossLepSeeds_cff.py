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

TkMuonPuppiHT6320 = l1tGTDoubleObjectCond.clone( #needs z0 with the puppivertex
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        regionsMinPt=cms.vdouble(6,6,6),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0) 
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTHtSum"),
        offlineMinScalarSumPt = cms.double(320),
    ),
)
pTkMuonPuppiHT6_320 = cms.Path(TkMuonPuppiHT6320)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonPuppiHT6_320")))


TkMuTriPuppiJetdRMaxDoubleJetdEtaMax = l1tGTQuadObjectCond.clone( #needs z0 between muon and puppivertex
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkMuonVLoose"),
        offlineMinPt = cms.double(12),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        minPt = cms.double(25) #safety cut
    ),
    collection3 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        minPt = cms.double(25) #safety cut
    ),
    collection4 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        minPt = cms.double(25) #safety cut
    ),
    correl12 = cms.PSet(
        maxDR = cms.double(0.4),
    ),
    correl34 = cms.PSet(
        maxDEta = cms.double(1.6)
    ),
)
pTkMuTriPuppiJet_12_40_dRMax_DoubleJet_dEtaMax = cms.Path(TkMuTriPuppiJetdRMaxDoubleJetdEtaMax)
algorithms.append(cms.PSet(expression=cms.string("pTkMuTriPuppiJet_12_40_dRMax_DoubleJet_dEtaMax")))

TkMuPuppiJetPuppiMet = l1tGTTripleObjectCond.clone( #needs z0 between muon and puppivertex
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        offlineMinPt = cms.double(110),
    ),
    collection3 = cms.PSet(
        object = gt_ref("l1tGTEtSum"),
        offlineMinPt = cms.double(120),
    ),
)
pTkMuPuppiJetPuppiMet_3_110_120 = cms.Path(TkMuPuppiJetPuppiMet)
algorithms.append(cms.PSet(expression=cms.string("pTkMuPuppiJetPuppiMet_3_110_120")))


DoubleTkMuPuppiJetPuppiMet = l1tGTQuadObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        offlineMinPt = cms.double(60),
    ),
    collection4 = cms.PSet(
        object = gt_ref("l1tGTEtSum"),
        offlineMinPt = cms.double(130),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
    )
)
pDoubleTkMuPuppiJetPuppiMet_3_3_60_130 = cms.Path(DoubleTkMuPuppiJetPuppiMet)
algorithms.append(cms.PSet(expression=cms.string("pDoubleTkMuPuppiJetPuppiMet_3_3_60_130")))


DoubleTkMuPuppiHT = l1tGTTripleObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        regionsMinPt=cms.vdouble(3,3,3),
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        maxPrimVertDz = cms.double(1),
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTtkMuonLoose"),
        regionsMinPt=cms.vdouble(3,3,3),
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        maxPrimVertDz = cms.double(1),
    ),
    collection3 = cms.PSet(
        object = gt_ref("l1tGTHtSum"),
        offlineMinScalarSumPt = cms.double(300),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
    )
)
pDoubleTkMuPuppiHT_3_3_300 = cms.Path(DoubleTkMuPuppiHT)

algorithms.append(cms.PSet(expression=cms.string("pDoubleTkMuPuppiHT_3_3_300")))


DoubleTkElePuppiHT = l1tGTTripleObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkElectronLowPt"),
        offlineMinPt = cms.double(8),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTtkElectronLowPt"),
        offlineMinPt = cms.double(8),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = cms.PSet(
        object = gt_ref("l1tGTHtSum"),
        offlineMinScalarSumPt = cms.double(390),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
    ),
)
pDoubleTkElePuppiHT_8_8_390 = cms.Path(DoubleTkElePuppiHT)

algorithms.append(cms.PSet(expression=cms.string("pDoubleTkElePuppiHT_8_8_390")))


TkEleIsoPuppiHT = l1tGTDoubleObjectCond.clone( #missing z0 between electron and puppivertex
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkIsoElectron"),
        minEta = cms.double(-2.1), # TBC WHY?
        maxEta = cms.double(2.1), # TBC WHY?
        offlineMinPt = cms.double(26),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTHtSum"),
        offlineMinScalarSumPt = cms.double(190),
    ),
)
pTkEleIsoPuppiHT_26_190 = cms.Path(TkEleIsoPuppiHT)
algorithms.append(cms.PSet(expression = cms.string("pTkEleIsoPuppiHT_26_190")))


TkElePuppiJetMinDR = l1tGTDoubleObjectCond.clone( #missing z0 between electron and puppivertex
    collection1 = cms.PSet(
        object = gt_ref("l1tGTtkElectron"),
        minEta = cms.double(-2.1), # TBC WHY?
        maxEta = cms.double(2.1), # TBC WHY?
        offlineMinPt = cms.double(28),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTsc4Jet"),
        minPt = cms.double(25) #safety cut
    ),
    minDR = cms.double(0.3)
)
pTkElePuppiJet_28_40_MinDR = cms.Path(TkElePuppiJetMinDR)
algorithms.append(cms.PSet(expression=cms.string("pTkElePuppiJet_28_40_MinDR")))

NNPuppiTauPuppiMet = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        object = gt_ref("l1tGTnnTau"),
        offlineMinPt = cms.double(55),
    ),
    collection2 = cms.PSet(
        object = gt_ref("l1tGTEtSum"),
        offlineMinPt = cms.double(190),
    ),
    
)
pNNPuppiTauPuppiMet_55_190 = cms.Path(NNPuppiTauPuppiMet)
algorithms.append(cms.PSet(expression=cms.string("pNNPuppiTauPuppiMet_55_190")))

