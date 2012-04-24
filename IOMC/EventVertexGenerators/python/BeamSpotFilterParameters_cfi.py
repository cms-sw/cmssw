import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

baseVtx = Realistic8TeVCollisionVtxSmearingParameters.clone()
newVtx = Realistic8TeV2012CollisionVtxSmearingParameters.clone()




