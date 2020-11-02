import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var

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
