import FWCore.ParameterSet.Config as cms

# Load BetafuncEvtVtxGenerator and read parameters from GT (SimBeamSpotObjectRcd)
from IOMC.EventVertexGenerators.BetafuncEvtVtxGenerator_cfi import BetafuncEvtVtxGenerator
VtxSmeared = BetafuncEvtVtxGenerator.clone(
    src = "generator:unsmeared",
    readDB = True
)

##
## Set the the vertex used for the simulation to the measured vertex for the tau embedding method
##
from Configuration.ProcessModifiers.tau_embedding_sim_cff import tau_embedding_sim
from TauAnalysis.MCEmbeddingTools.Simulation_GEN_cfi import tau_embedding_vtx_corrected_to_input
tau_embedding_sim.toReplaceWith(VtxSmeared, tau_embedding_vtx_corrected_to_input)