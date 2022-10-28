import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var,ExtVar
from PhysicsTools.NanoAOD.nano_eras_cff import *

genVertexTable = cms.EDProducer("SimpleXYZPointFlatTableProducer",
    src = cms.InputTag("genParticles:xyz0"),
    cut = cms.string(""), 
    name= cms.string("GenVtx"),
    doc = cms.string("Gen vertex"),
    singleton = cms.bool(True), 
    extension = cms.bool(False), 
    variables = cms.PSet(
         x = Var("X", float, doc="gen vertex x", precision=10),
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

genVertexTablesTask = cms.Task(genVertexTable,genVertexT0Table)

# GenVertex only stored in newer MiniAOD
(run2_nanoAOD_92X | run2_miniAOD_80XLegacy | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1).toReplaceWith(genVertexTablesTask, cms.Task())
