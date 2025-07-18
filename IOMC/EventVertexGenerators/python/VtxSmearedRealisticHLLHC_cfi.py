import FWCore.ParameterSet.Config as cms

# Load HLLHCEvtVtxGenerator and read parameters from GT (SimBeamSpotHLLHCObjectsRcd)
from IOMC.EventVertexGenerators.HLLHCEvtVtxGenerator_cfi import HLLHCEvtVtxGenerator
VtxSmeared = HLLHCEvtVtxGenerator.clone(
    src = "generator:unsmeared",
    readDB = True
)
