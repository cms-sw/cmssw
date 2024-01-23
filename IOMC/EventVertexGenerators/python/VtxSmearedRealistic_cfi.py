import FWCore.ParameterSet.Config as cms

# Load BetafuncEvtVtxGenerator and read parameters from GT (SimBeamSpotObjectRcd)
from IOMC.EventVertexGenerators.BetafuncEvtVtxGenerator_cfi import BetafuncEvtVtxGenerator
VtxSmeared = BetafuncEvtVtxGenerator.clone(
    src = "generator:unsmeared",
    readDB = True
)
