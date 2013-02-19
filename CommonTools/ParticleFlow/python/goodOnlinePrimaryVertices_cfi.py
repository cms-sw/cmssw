import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

goodOnlinePrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
)
