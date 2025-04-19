import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
#from PhysicsTools.NanoAOD.vertices_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

pVertexTable = cms.EDProducer("PVertexBPHTable",
    pvSrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    goodPvCut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"), 
    pvName = cms.string("PVtx")
)
