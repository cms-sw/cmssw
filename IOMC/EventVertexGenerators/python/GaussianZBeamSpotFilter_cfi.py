import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.BeamSpotFilterParameters_cfi import *
simBeamSpotFilter = cms.EDFilter("GaussianZBeamSpotFilter",
    src = cms.InputTag("generator"),
    baseSZ = baseVtx.SigmaZ,
    baseZ0 = baseVtx.Z0,
    newSZ = newVtx.SigmaZ,
    newZ0 = newVtx.Z0
)



