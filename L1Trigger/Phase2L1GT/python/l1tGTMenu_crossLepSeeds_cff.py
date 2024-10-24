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

from L1Trigger.Phase2L1GT.menuConstants import *

TkMuonTkIsoEle720 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(7,7,7), # no scaling used below 8 GeV
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(20, "L1tkElectron","Iso"),
        regionsMaxRelIsolationPt = getObjectISOs("L1tkElectron","Iso"),
    ),
    maxDz = cms.double(1),
)
pTkMuonTkIsoEle7_20 = cms.Path(TkMuonTkIsoEle720)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonTkIsoEle7_20")))

TkMuonTkEle723 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(7,7,7), # no scaling used below 8 GeV
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(23, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIso"),
    ),
    maxDz = cms.double(1),
)
pTkMuonTkEle7_23 = cms.Path(TkMuonTkEle723)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonTkEle7_23")))

TkEleTkMuon1020 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(36, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIso"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt = getObjectThrs(20, "L1gmtTkMuon","VLoose"),
        qualityFlags = getObjectIDs("L1gmtTkMuon","VLoose"),
    ),
    maxDz = cms.double(1),
)
pTkEleTkMuon10_20 = cms.Path(TkEleTkMuon1020)
algorithms.append(cms.PSet(expression = cms.string("pTkEleTkMuon10_20")))

TkMuonDoubleTkEle61717 = l1tGTTripleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(6,6,6),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(17, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIsoLowPt"),
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(17, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIsoLowPt"),
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
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(5,5,5),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(5,5,5),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(9, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIsoLowPt"),

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
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt = getObjectThrs(18, "L1gmtTkMuon","VLoose"),
        qualityFlags = getObjectIDs("L1gmtTkMuon","VLoose"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0) # primary vertex index (choose 0) 
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt = getObjectThrs(42, "L1nnPuppiTau","default"),
        minQualityScore = getObjectIDs("L1nnPuppiTau","default")
    ),
)
pPuppiTauTkMuon42_18 = cms.Path(PuppiTauTkMuon4218)
algorithms.append(cms.PSet(expression = cms.string("pPuppiTauTkMuon42_18")))

PuppiTauTkIsoEle4522 = l1tGTDoubleObjectCond.clone(  ###NB We need puppivertex here
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(22, "L1tkElectron","Iso"),
        regionsMaxRelIsolationPt = getObjectISOs("L1tkElectron","Iso"),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0) # primary vertex index (choose 0) 
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt = getObjectThrs(45, "L1nnPuppiTau","default"),
        minQualityScore = getObjectIDs("L1nnPuppiTau","default")
    ),
)
pPuppiTauTkIsoEle45_22 = cms.Path(PuppiTauTkIsoEle4522)
algorithms.append(cms.PSet(expression = cms.string("pPuppiTauTkIsoEle45_22")))


