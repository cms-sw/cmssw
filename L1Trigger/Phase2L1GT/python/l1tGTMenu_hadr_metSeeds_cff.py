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
    tag =  cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
    regionsMinPt=cms.vdouble(160,108) 
)
pSinglePuppiJet230 = cms.Path(SinglePuppiJet230)
algorithms.append(cms.PSet(expression = cms.string("pSinglePuppiJet230")))

DoublePuppiJet112112 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(70,51)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(70,51)
    ),
    maxDEta = cms.double(1.6),
)
pDoublePuppiJet112_112 = cms.Path(DoublePuppiJet112112)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiJet112_112")))

DoublePuppiJet16035Mass620 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-5),
        maxEta = cms.double(5),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(107,74)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-5),
        maxEta = cms.double(5),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25,25) #safety cut, in truth 11 and 13 GeV
    ),
    minInvMass = cms.double(620),
)
pDoublePuppiJet160_35_mass620 = cms.Path(DoublePuppiJet16035Mass620)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiJet160_35_mass620")))


PuppiHT450 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    minScalarSumPt = cms.double(371)
)
pPuppiHT450 = cms.Path(PuppiHT450)
algorithms.append(cms.PSet(expression = cms.string("pPuppiHT450")))

PuppiMHT140 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    minPt = cms.double(133)
)
pPuppiMHT140 = cms.Path(PuppiMHT140)
algorithms.append(cms.PSet(expression = cms.string("pPuppiMHT140")))

PuppiMET200 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2EtSum"),
    minPt = cms.double(93)
)
pPuppiMET200 = cms.Path(PuppiMET200)
algorithms.append(cms.PSet(expression = cms.string("pPuppiMET200")))

QuadJet70554040 = l1tGTQuadObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(38,30)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(27,25.0)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25.0,25.0)
    ),
    collection4 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(25.0,25.0)
    ),

)
pQuadJet70_55_40_40 = cms.Path(QuadJet70554040)

PuppiHT400 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2HtSum"),
    minScalarSumPt = cms.double(325) 
)
pPuppiHT400 = cms.Path(PuppiHT400)


algorithms.append(cms.PSet(name=cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
                       expression=cms.string("pPuppiHT400 and pQuadJet70_55_40_40")))

