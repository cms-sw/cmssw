import FWCore.ParameterSet.Config as cms

# Load GaussEvtVtxGenerator and read parameters from GT (SimBeamSpotObjectRcd)
from IOMC.EventVertexGenerators.GaussEvtVtxGenerator_cfi import GaussEvtVtxGenerator
VtxSmeared = GaussEvtVtxGenerator.clone(
    src = "generator:unsmeared",
    readDB = True
)
