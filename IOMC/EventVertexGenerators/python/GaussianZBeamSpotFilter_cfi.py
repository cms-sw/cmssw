import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.BeamSpotFilterParameters_cfi import baseVtx,newVtx
simBeamSpotFilter = cms.EDFilter("GaussianZBeamSpotFilter",
    src = cms.InputTag("generatorSmeared"),
    baseSZ = baseVtx.SigmaZ,
    baseZ0 = baseVtx.Z0,
    newSZ = newVtx.SigmaZ,
    newZ0 = newVtx.Z0
)



