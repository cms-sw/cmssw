import FWCore.ParameterSet.Config as cms

goodOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!obj.isFake() && obj.ndof() >= 4.0 && std::abs(obj.z()) <= 24.0 && abs(obj.position().Rho()) <= 2.0"),
   filter = cms.bool(False)
)

