import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var,ExtVar
from PhysicsTools.NanoAOD.globalVariablesTableProducer_cfi import globalVariablesTableProducer
from PhysicsTools.NanoAOD.simpleXYZPointFlatTableProducer_cfi import simpleXYZPointFlatTableProducer

genVertexTable = simpleXYZPointFlatTableProducer.clone(
    src = cms.InputTag("genParticles:xyz0"),
    name= cms.string("GenVtx"),
    doc = cms.string("Gen vertex"),
    variables = cms.PSet(
         x = Var("X", float, doc="gen vertex x", precision=10),
         y = Var("Y", float, doc="gen vertex y", precision=10),
         z = Var("Z", float, doc="gen vertex z", precision=16),
    )
)

genVertexT0Table = globalVariablesTableProducer.clone(
    name = cms.string("GenVtx"),
    extension = cms.bool(True),
    variables = cms.PSet(
        t0 = ExtVar( cms.InputTag("genParticles:t0"), "float", doc = "gen vertex t0", precision=12),
    )
)

genVertexTablesTask = cms.Task(genVertexTable,genVertexT0Table)
