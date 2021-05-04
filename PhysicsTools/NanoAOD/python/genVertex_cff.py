import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var,ExtVar

genVertexTable = cms.EDProducer("SimpleXYZPointFlatTableProducer",
    src = cms.InputTag("genParticles:xyz0"),
    cut = cms.string(""), 
    name= cms.string("GenVtx"),
    doc = cms.string("Gen vertex"),
    singleton = cms.bool(True), 
    extension = cms.bool(False), 
    variables = cms.PSet(
         x  = Var("X", float, doc="gen vertex x", precision=10),
         y = Var("Y", float, doc="gen vertex y", precision=10),
         z = Var("Z", float, doc="gen vertex z", precision=16),
    ) 
)

genVertexT0Table = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string("GenVtx"),
    extension = cms.bool(True), 
    variables = cms.PSet(
        t0 = ExtVar( cms.InputTag("genParticles:t0"), "float", doc = "gen vertex t0", precision=12),
    )
)

genVertexTables = cms.Sequence(genVertexTable+genVertexT0Table)
