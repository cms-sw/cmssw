import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

METTable = cms.EDProducer(
    "SimpleMETFlatTableProducer",
    src = cms.InputTag("hltPFPuppiMETTypeOne"),
    name = cms.string("hltPFPuppiMET"),
    doc = cms.string("HLT PF Puppi MET (TypeOne) information"),
    extension = cms.bool(False),
    skipNonExistingSrc = cms.bool(True),
    singleton = cms.bool(True),
    variables = cms.PSet(
        pt = Var("sumEt", "float", doc = "MET p_T (GeV)"),
    ),
)

HTTable = cms.EDProducer(
    "SimpleMETFlatTableProducer",
    src = cms.InputTag("hltPFPuppiMHT"),
    name = cms.string("hltPFPuppiHT"),
    doc = cms.string("HLT PF Puppi HT information"),
    extension = cms.bool(False),
    skipNonExistingSrc = cms.bool(True),
    singleton = cms.bool(True),
    variables = cms.PSet(
        pt = Var("sumEt", "float", doc = "HT p_T (GeV)"),
    ),
)
