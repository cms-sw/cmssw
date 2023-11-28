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

####### JET, MET, HT ###########

SinglePuppiJet230 = l1tGTSingleObjectCond.clone(
    tag =  cms.InputTag("l1tGTProducer", "CL2Jets"),
    #minPt = cms.double(164.9),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
    regionsMinPt=cms.vdouble(160.5,108.3) 
)
pSinglePuppiJet230 = cms.Path(SinglePuppiJet230)
algorithms.append(cms.PSet(expression = cms.string("pSinglePuppiJet230")))

PuppiHT450 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    minScalarSumPt = cms.double(372.9)
)
pPuppiHT450 = cms.Path(PuppiHT450)
algorithms.append(cms.PSet(expression = cms.string("pPuppiHT450")))


PuppiMET200 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),
    minPt = cms.double(93.1)
)
pPuppiMET200 = cms.Path(PuppiMET200)
algorithms.append(cms.PSet(expression = cms.string("pPuppiMET200")))

QuadJet70554040 = l1tGTQuadObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Jets"),
        #minPt = cms.double(41.9),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(42.0,32.7)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Jets"),
        #minPt = cms.double(30.3),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(26.7,25.0)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Jets"),
        #minPt = cms.double(18.8),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25.0,25.0)
    ),
    collection4 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Jets"),
        #minPt = cms.double(18.8),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25.0,25.0)
    ),

)
pQuadJet70_55_40_40 = cms.Path(QuadJet70554040)

PuppiHT400 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    minScalarSumPt = cms.double(326.9) 
)
pPuppiHT400 = cms.Path(PuppiHT400)


algorithms.append(cms.PSet(name=cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
                       expression=cms.string("pPuppiHT400 and pQuadJet70_55_40_40")))

