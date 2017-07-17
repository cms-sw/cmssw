import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseForTripletsByCellularAutomaton import customiseNewSeeding

def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)


def customiseForTripletsHLTPixelTracksByCellularAutomaton(process):
    for module in producers_by_type(process, "PixelTrackProducer"):
        tripletModuleName = module.SeedingHitSets.getModuleLabel()
        tripletModule = getattr(process, tripletModuleName)
        if tripletModule._TypedParameterizable__type in ["PixelTripletHLTEDProducer", "PixelTripletLargeTipEDProducer"]:
            customiseNewSeeding(process, tripletModule)

    return process
