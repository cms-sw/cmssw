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

TkMuonPuppiHT6320 = l1tGTDoubleObjectCond.clone( #needs z0 with the puppivertex
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(6,6,6),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0) 
        qualityFlags = cms.uint32(0b0001)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
        minScalarSumPt = cms.double(251) 
    ),
)
pTkMuonPuppiHT6_320 = cms.Path(TkMuonPuppiHT6320)
algorithms.append(cms.PSet(expression = cms.string("pTkMuonPuppiHT6_320")))


TkMuTriPuppiJetdRMaxDoubleJetdEtaMax = l1tGTQuadObjectCond.clone( #needs z0 between muon and puppivertex
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(10,10,11), 
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25,25), #safety cut, actually 15 and 16
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25,25), #safety cut, actually 15 and 16regionsMinPt=cms.vdouble(25.0,25.0)
    ),
    collection4 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25,25), #safety cut, actually 15 and 16
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
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(69,50)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),        
        minPt = cms.double(38)
    ),
    
)
pTkMuPuppiJetPuppiMet_3_110_120 = cms.Path(TkMuPuppiJetPuppiMet)

algorithms.append(cms.PSet(expression=cms.string("pTkMuPuppiJetPuppiMet_3_110_120")))


DoubleTkMuPuppiJetPuppiMet = l1tGTQuadObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(30,25)
    ),
    collection4 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),        
        minPt = cms.double(45)
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
    )
)
pDoubleTkMuPuppiJetPuppiMet_3_3_60_130 = cms.Path(DoubleTkMuPuppiJetPuppiMet)

algorithms.append(cms.PSet(expression=cms.string("pDoubleTkMuPuppiJetPuppiMet_3_3_60_130")))


DoubleTkMuPuppiHT = l1tGTTripleObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(3,3,3),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
        qualityFlags = cms.uint32(0b0001)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
        minScalarSumPt = cms.double(232) 
    ),
    correl12 = cms.PSet(
        minDR = cms.double(0),
    )
)
pDoubleTkMuPuppiHT_3_3_300 = cms.Path(DoubleTkMuPuppiHT)

algorithms.append(cms.PSet(expression=cms.string("pDoubleTkMuPuppiHT_3_3_300")))


DoubleTkElePuppiHT = l1tGTTripleObjectCond.clone( #needs z0 between puppivertex and muon
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(6,6),
        regionsQualityFlags=cms.vuint32(0b0010,0b0000),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(6,6),
        regionsQualityFlags=cms.vuint32(0b0010,0b0000), 
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
        minScalarSumPt = cms.double(316) 
    ),
)
pDoubleTkElePuppiHT_8_8_390 = cms.Path(DoubleTkElePuppiHT)

algorithms.append(cms.PSet(expression=cms.string("pDoubleTkElePuppiHT_8_8_390")))


TkEleIsoPuppiHT = l1tGTDoubleObjectCond.clone( #missing z0 between electron and puppivertex
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(21,20), #no qualities as online cut below 25 in the endcap
        regionsMaxRelIsolationPt = cms.vdouble(0.13,0.28),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
        minScalarSumPt = cms.double(131) 
    ),
)
pTkEleIsoPuppiHT_26_190 = cms.Path(TkEleIsoPuppiHT)
algorithms.append(cms.PSet(expression = cms.string("pTkEleIsoPuppiHT_26_190")))


TkElePuppiJetMinDR = l1tGTDoubleObjectCond.clone( #missing z0 between electron and puppivertex
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minEta = cms.double(-2.1),
        maxEta = cms.double(2.1),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(23,22),
        regionsQualityFlags=cms.vuint32(0b0010,0b0000),
        maxPrimVertDz = cms.double(1), # in cm
        primVertex = cms.uint32(0), # primary vertex index (choose 0)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25,25), #safety cut, actually 15,16
    ),
    minDR = cms.double(0.3)
)
pTkElePuppiJet_28_40_MinDR = cms.Path(TkElePuppiJetMinDR)

algorithms.append(cms.PSet(expression=cms.string("pTkElePuppiJet_28_40_MinDR")))



NNPuppiTauPuppiMet = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(30,22),
        minQualityScore = cms.uint32(225),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),        
        minPt = cms.double(86)
    ),
    
)
pNNPuppiTauPuppiMet_55_190 = cms.Path(NNPuppiTauPuppiMet)

algorithms.append(cms.PSet(expression=cms.string("pNNPuppiTauPuppiMet_55_190")))

