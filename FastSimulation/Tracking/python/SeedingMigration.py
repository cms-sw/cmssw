import FWCore.ParameterSet.Config as cms
import copy

def _copy(old, new, skip=[]):
    skipSet = set(skip)
    for key in old.parameterNames_():
        if key not in skipSet:
            setattr(new, key, copy.deepcopy(getattr(old, key)))

def _hitSetProducerToFactoryPSet(producer):
    _map = {
        "PixelTripletHLTEDProducer": "PixelTripletHLTGenerator",
        "PixelTripletLargeTipEDProducer": "PixelTripletLargeTipGenerator",
        "MultiHitFromChi2EDProducer": "MultiHitGeneratorFromChi2",
        "CAHitTripletEDProducer": "CAHitTripletGenerator",
        "CAHitQuadrupletEDProducer": "CAHitQuadrupletGenerator",
        "CAHitNtupletHeterogeneousEDProducer": "CAHitQuadrupletGenerator",
    }
    ret = cms.PSet()
    _copy(producer, ret)
    ret.ComponentName = cms.string(_map[producer._TypedParameterizable__type]);
    return ret
