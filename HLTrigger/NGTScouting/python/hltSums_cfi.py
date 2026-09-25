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
        pt      = Var("pt",             "float", doc = "MET p_T (GeV)"),
        phi     = Var("phi",            "float", doc = "MET phi"),
        px      = Var("px",             "float", doc = "MET x component (GeV)"),
        py      = Var("py",             "float", doc = "MET y component (GeV)"),
        sumEt   = Var("sumEt",          "float", doc = "scalar sum of E_T over all objects (GeV)"),
        mEtSig  = Var("mEtSig",         "float", doc = "MET / sqrt(sumEt)"),
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
        pt      = Var("sumEt", "float", doc = "HT: scalar sum of jet p_T (GeV)"),
        mhtPt   = Var("pt",    "float", doc = "MHT p_T (GeV)"),
        mhtPhi  = Var("phi",   "float", doc = "MHT phi"),
        mhtPx   = Var("px",    "float", doc = "MHT x component (GeV)"),
        mhtPy   = Var("py",    "float", doc = "MHT y component (GeV)"),
        mhtSig  = Var("mEtSig","float", doc = "MHT / sqrt(HT)"),
    ),
)
