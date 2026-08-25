import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2
from Configuration.Eras.Modifier_phase2_fastSim_cff import phase2_fastSim
# The MC-truth graph accumulator reads full-sim g4SimHits during classic mixing, which
# FastSim does not provide, so enableTruth is dropped from every FastSim era here.
from Configuration.ProcessModifiers.enableTruth_cff import enableTruth

def fastSimPhase2(obj):
    return cms.ModifierChain(
        obj.copyAndExclude([run3_GEM, phase2_muon, phase2_GEM, phase2_trigger, trackingMkFitProdPhase2, enableTruth]),
        phase2_fastSim,
    )
