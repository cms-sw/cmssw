import FWCore.ParameterSet.Config as cms

def _regionProducerToFactoryPSet(producer):
    _map = {
        "GlobalTrackinRegionFromBeamSpotEDProducer": "GlobalRegionProducerFromBeamSpot",
    }
    return cms.PSet(
        ComponentName = cms.string(_map[producer._TypedParameterizable__type]),
        RegionPSet = producer.RegionPSet.clone(),
    )

def _copy(old, new, skip=[]):
    skipSet = set(skip)
    for key in old.parameterNames_():
        if key not in skipSet:
            setattr(new, key, getattr(old, key))

def _hitSetProducerToFactoryPSet(producer):
    _map = {
        "PixelTripletHLTEDProducer": "PixelTripletHLTGenerator",
        "PixelTripletLargeTipEDProducer": "PixelTripletLargeTipGenerator",
    }
    ret = cms.PSet()
    _copy(producer, ret)
    ret.ComponentName = cms.string(_map[producer._TypedParameterizable__type]);
    return ret
