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

####### JET, MET, HT ###########

SinglePuppiJet230 = l1tGTSingleObjectCond.clone(
    l1tGTsc4Jet.clone(),
    regionsMinPt = get_object_thrs(230, "CL2JetsSC4", "default"),
)
pSinglePuppiJet230 = cms.Path(SinglePuppiJet230)
algorithms.append(cms.PSet(expression = cms.string("pSinglePuppiJet230")))

DoublePuppiJet112112 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(112, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(112, "CL2JetsSC4", "default"),
    ),
    maxDEta = cms.double(1.6),
)
pDoublePuppiJet112_112 = cms.Path(DoublePuppiJet112112)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiJet112_112")))

DoublePuppiJet16035Mass620 = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(160, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet_er5.clone(
        regionsMinPt = get_object_thrs(35, "CL2JetsSC4", "default"),
    ),
    minInvMass = cms.double(620),
)
pDoublePuppiJet160_35_mass620 = cms.Path(DoublePuppiJet16035Mass620)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiJet160_35_mass620")))


PuppiHT450 = l1tGTSingleObjectCond.clone(
    l1tGTHtSum.clone(),
    minScalarSumPt = get_object_thrs(450, "CL2HtSum", "HT"),
)
pPuppiHT450 = cms.Path(PuppiHT450)
algorithms.append(cms.PSet(expression = cms.string("pPuppiHT450")))

PuppiMHT140 = l1tGTSingleObjectCond.clone(
    l1tGTHtSum.clone(),
    minPt = get_object_thrs(140, "CL2HtSum", "MHT"),
)
pPuppiMHT140 = cms.Path(PuppiMHT140)
algorithms.append(cms.PSet(expression = cms.string("pPuppiMHT140")))

PuppiMET200 = l1tGTSingleObjectCond.clone(
    l1tGTEtSum.clone(),
    minPt = get_object_thrs(200, "CL2EtSum", "default"),
)
pPuppiMET200 = cms.Path(PuppiMET200)
algorithms.append(cms.PSet(expression = cms.string("pPuppiMET200")))

QuadJet70554040 = l1tGTQuadObjectCond.clone(
    collection1 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(70, "CL2JetsSC4", "default"),
    ),
    collection2 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(55, "CL2JetsSC4", "default"),
    ),
    collection3 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(40, "CL2JetsSC4", "default"),
    ),
    collection4 = l1tGTsc4Jet.clone(
        regionsMinPt = get_object_thrs(40, "CL2JetsSC4", "default"),
    ),

)
pQuadJet70_55_40_40 = cms.Path(QuadJet70554040)

PuppiHT400 = l1tGTSingleObjectCond.clone(
    l1tGTHtSum.clone(),
    minScalarSumPt = get_object_thrs(400, "CL2HtSum", "HT"),
)
pPuppiHT400 = cms.Path(PuppiHT400)


algorithms.append(cms.PSet(name=cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
                       expression=cms.string("pPuppiHT400 and pQuadJet70_55_40_40")))

