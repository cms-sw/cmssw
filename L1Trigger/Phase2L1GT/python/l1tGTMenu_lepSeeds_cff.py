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

####### MUON SEEDS ###########

#        regionsAbsEtaLowerBounds=cms.vdouble(0,1.2,3),
#        regionsMinPt=cms.vdouble(12,14,15)


SingleTkMuon22 = l1tGTSingleObjectCond.clone(
    tag =  cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
    regionsMinPt = getObjectThrs(22, "L1gmtTkMuon","VLoose"),
    qualityFlags = getObjectIDs("L1gmtTkMuon","VLoose"),
)
pSingleTkMuon22 = cms.Path(SingleTkMuon22)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkMuon22")))

DoubleTkMuon157 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt = getObjectThrs(15, "L1gmtTkMuon","VLoose"),
        qualityFlags = getObjectIDs("L1gmtTkMuon","VLoose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(7,7,7), # no scaling used below 8 GeV
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    maxDz = cms.double(1),
    minDR = cms.double(0),
)
pDoubleTkMuon15_7 = cms.Path(DoubleTkMuon157)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon15_7")))

TripleTkMuon533 = l1tGTTripleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(5), # no scaling used below 8 GeV
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qualityFlags = getObjectIDs("L1gmtTkMuon","Loose"),
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1)
    ),
    correl13 = cms.PSet(
        minDR = cms.double(0),
        maxDz = cms.double(1)
    ),
    correl23 = cms.PSet(
        minDR = cms.double(0),
    )
)
pTripleTkMuon5_3_3 = cms.Path(TripleTkMuon533)
algorithms.append(cms.PSet(expression = cms.string("pTripleTkMuon5_3_3")))

####### EG and PHO seeds ###########

SingleEGEle51 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt = getObjectThrs(51, "L1tkPhoton","Iso"),
    regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
)
pSingleEGEle51 = cms.Path(SingleEGEle51) 
algorithms.append(cms.PSet(expression = cms.string("pSingleEGEle51")))

DoubleEGEle3724 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(37, "L1tkPhoton","Iso"),
        regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(24, "L1tkPhoton","Iso"), 
        regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
    ),
    minDR = cms.double(0.1),
)
pDoubleEGEle37_24 = cms.Path(DoubleEGEle3724)
algorithms.append(cms.PSet(expression = cms.string("pDoubleEGEle37_24")))

IsoTkEleEGEle2212 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(22, "L1tkElectron","Iso"),
        regionsMaxRelIsolationPt = getObjectISOs("L1tkElectron","Iso"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(12, "L1tkPhoton","Iso"),
        regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
    ),
    minDR = cms.double(0.1),
)
pIsoTkEleEGEle22_12 = cms.Path(IsoTkEleEGEle2212)
algorithms.append(cms.PSet(expression = cms.string("pIsoTkEleEGEle22_12")))

SingleTkEle36 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt = getObjectThrs(36, "L1tkElectron","NoIso"),
    regionsQualityFlags = getObjectIDs("L1tkElectron","NoIso"),
    # regionsMaxRelIsolationPt = getObjectISOs("L1tkElectron","NoIso"),

)
pSingleTkEle36 = cms.Path(SingleTkEle36) 
algorithms.append(cms.PSet(expression = cms.string("pSingleTkEle36")))

SingleIsoTkEle28 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt = getObjectThrs(28, "L1tkElectron","Iso"),
    # regionsQualityFlags = getObjectIDs("L1tkElectron","Iso"),
    regionsMaxRelIsolationPt = getObjectISOs("L1tkElectron","Iso"),
)
pSingleIsoTkEle28 = cms.Path(SingleIsoTkEle28) 
algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28")))

SingleIsoTkPho36 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt = getObjectThrs(36, "L1tkPhoton","Iso"),
    regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
    regionsMaxRelIsolationPt = getObjectISOs("L1tkPhoton","Iso"),
)
pSingleIsoTkPho36 = cms.Path(SingleIsoTkPho36) 

algorithms.append(cms.PSet(expression=cms.string("pSingleIsoTkPho36")))

DoubleTkEle2512 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(25, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIsoLowPt"),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(12, "L1tkElectron","NoIso"),
        regionsQualityFlags = getObjectIDs("L1tkElectron","NoIsoLowPt"),
    ),
    maxDz = cms.double(1),
)
pDoubleTkEle25_12 = cms.Path(DoubleTkEle2512)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle25_12")))

DoubleIsoTkPho2212 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(22, "L1tkPhoton","Iso"),
        regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
        regionsMaxRelIsolationPt = cms.vdouble(
            getObjectISOs("L1tkPhoton","Iso")
        )
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt = getObjectThrs(12, "L1tkPhoton","Iso"),
        regionsQualityFlags = getObjectIDs("L1tkPhoton","Iso"),
        regionsMaxRelIsolationPt = getObjectISOs("L1tkPhoton","Iso")
    ),
)
pDoubleIsoTkPho22_12 = cms.Path(DoubleIsoTkPho2212)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTkPho22_12")))

DoublePuppiTau5252 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt = getObjectThrs(52, "L1nnPuppiTau","default"),
        minQualityScore = getObjectIDs("L1nnPuppiTau","default")
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt = getObjectThrs(52, "L1nnPuppiTau","default"),
        minQualityScore = getObjectIDs("L1nnPuppiTau","default")
    ),
    minDR = cms.double(0.5),
)
pDoublePuppiTau52_52 = cms.Path(DoublePuppiTau5252)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiTau52_52")))

