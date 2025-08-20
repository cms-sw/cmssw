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

####### MUON SEEDS ###########

SingleTkMuon22 = l1tGTSingleObjectCond.clone(
    l1tGTtkMuonVLoose.clone(),
    regionsMinPt = get_object_thrs(22, "GMTTkMuons","VLoose"),
)
pSingleTkMuon22 = cms.Path(SingleTkMuon22)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkMuon22")))

DoubleTkMuon157 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuonVLoose.clone(
        regionsMinPt = get_object_thrs(15, "GMTTkMuons","VLoose"),
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        regionsMinPt = cms.vdouble(7,7,7), # no scaling used below 8 GeV
    ),
    maxDz = cms.double(1),
    minDR = cms.double(0),
)
pDoubleTkMuon15_7 = cms.Path(DoubleTkMuon157)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon15_7")))

TripleTkMuon533 = l1tGTTripleObjectCond.clone(
    collection1 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(5), # no scaling used below 8 GeV
    ),
    collection2 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(3),
    ),
    collection3 = l1tGTtkMuonLoose.clone(
        minPt = cms.double(3),
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
    l1tGTtkPhoton.clone(),
    regionsMinPt = get_object_thrs(51, "CL2Photons","Iso"),
)
pSingleEGEle51 = cms.Path(SingleEGEle51) 
algorithms.append(cms.PSet(expression = cms.string("pSingleEGEle51")))

DoubleEGEle3724 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoPhoton.clone(
        regionsMinPt = get_object_thrs(37, "CL2Photons","Iso"),
    ),
    collection2 = l1tGTtkIsoPhoton.clone(
        regionsMinPt = get_object_thrs(24, "CL2Photons","Iso"), 
    ),
    minDR = cms.double(0.1),
)
pDoubleEGEle37_24 = cms.Path(DoubleEGEle3724)
algorithms.append(cms.PSet(expression = cms.string("pDoubleEGEle37_24")))

IsoTkEleEGEle2212 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoElectron.clone(
        regionsMinPt = get_object_thrs(22, "CL2Electrons","Iso"),
    ),
    collection2 = l1tGTtkIsoPhoton.clone(
        regionsMinPt = get_object_thrs(12, "CL2Photons","Iso"),
    ),
    minDR = cms.double(0.1),
)
pIsoTkEleEGEle22_12 = cms.Path(IsoTkEleEGEle2212)
algorithms.append(cms.PSet(expression = cms.string("pIsoTkEleEGEle22_12")))

SingleTkEle36 = l1tGTSingleObjectCond.clone(
    l1tGTtkElectron.clone(),
    regionsMinPt = get_object_thrs(36, "CL2Electrons","NoIso"),
)
pSingleTkEle36 = cms.Path(SingleTkEle36) 
algorithms.append(cms.PSet(expression = cms.string("pSingleTkEle36")))

SingleIsoTkEle28 = l1tGTSingleObjectCond.clone(
    l1tGTtkIsoElectron.clone(),
    regionsMinPt = get_object_thrs(28, "CL2Electrons","Iso"),
)
pSingleIsoTkEle28 = cms.Path(SingleIsoTkEle28) 
algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28")))

SingleIsoTkPho36 = l1tGTSingleObjectCond.clone(
    l1tGTtkIsoPhoton.clone(),
    regionsMinPt = get_object_thrs(36, "CL2Photons","Iso"),
)
pSingleIsoTkPho36 = cms.Path(SingleIsoTkPho36) 

algorithms.append(cms.PSet(expression=cms.string("pSingleIsoTkPho36")))

DoubleTkEle2512 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkElectronLowPt.clone(
        regionsMinPt = get_object_thrs(25, "CL2Electrons","NoIso"),
    ),
    collection2 = l1tGTtkElectronLowPt.clone(
        regionsMinPt = get_object_thrs(12, "CL2Electrons","NoIso"),
    ),
    maxDz = cms.double(1),
)
pDoubleTkEle25_12 = cms.Path(DoubleTkEle2512)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle25_12")))

DoubleIsoTkPho2212 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkIsoPhoton.clone(
        regionsMinPt = get_object_thrs(22, "CL2Photons","Iso"),
    ),
    collection2 = l1tGTtkIsoPhoton.clone(
        regionsMinPt = get_object_thrs(12, "CL2Photons","Iso"),
    ),
)
pDoubleIsoTkPho22_12 = cms.Path(DoubleIsoTkPho2212)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTkPho22_12")))

DoublePuppiTau5252 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTnnTau.clone(
        regionsMinPt = get_object_thrs(52, "CL2Taus","default"),
    ),
    collection2 = l1tGTnnTau.clone(
        regionsMinPt = get_object_thrs(52, "CL2Taus","default"),
    ),
    minDR = cms.double(0.5),
)
pDoublePuppiTau52_52 = cms.Path(DoublePuppiTau5252)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiTau52_52")))

