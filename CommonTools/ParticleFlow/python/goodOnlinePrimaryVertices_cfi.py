import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

goodOnlinePrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = 4.0, maxZ = 24.0 ),
    src = cms.InputTag('offlinePrimaryVertices')
)
