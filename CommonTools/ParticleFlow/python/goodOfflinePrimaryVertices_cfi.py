import FWCore.ParameterSet.Config as cms

goodOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 4.0 && abs(z) <= 24.0 && abs(position.Rho) <= 2.0"),
   filter = cms.bool(False)
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    goodOfflinePrimaryVertices,
    src = cms.InputTag("offlinePrimaryVertices4D"),
)
