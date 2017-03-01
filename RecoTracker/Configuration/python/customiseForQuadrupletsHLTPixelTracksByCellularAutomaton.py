import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseForQuadrupletsByCellularAutomaton import customiseNewSeeding

def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)
    
def customiseForQuadrupletsHLTPixelTracksByCellularAutomaton(process):
    for module in producers_by_type(process, "PixelTrackProducer"):
        quadrupletModuleName = module.SeedingHitSets.getModuleLabel()
        quadrupletModule = getattr(process, quadrupletModuleName)
        if quadrupletModule._TypedParameterizable__type == "PixelQuadrupletEDProducer":
            customiseNewSeeding(process, quadrupletModule)

    return process
