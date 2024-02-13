import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.l1trig_cff import *


gtVtxTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src = cms.InputTag('l1tGTProducer','GTTPrimaryVert'),
    cut = cms.string(""),
    name = cms.string("L1GTVertex"),
    doc = cms.string("GT GTT Vertices"),
    singleton = cms.bool(False), # the number of entries is variable
    variables = cms.PSet(
        z0 = Var("vz",float, doc = "primary vertex position z coordinate"),
        # sumPt = Var("hwSum_pT_pv_toInt()*0.25",float, doc = "sum pt of tracks"),
        ## hw vars
        hwZ0 = Var("hwZ0_toInt()",int, doc = "HW primary vertex position z coordinate"),
        # hwSum_pT_pv = Var("hwSum_pT_pv_toInt()",int, doc = "HW sum pt of tracks"),
    )
)

p2GTL1TablesTask = cms.Task(gtVtxTable)


