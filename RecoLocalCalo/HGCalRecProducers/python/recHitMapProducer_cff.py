import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer as recHitMapProducer_

recHitMapProducer = recHitMapProducer_.clone()

hits = ["HGCalRecHit:HGCEERecHits",
        "HGCalRecHit:HGCHEFRecHits",
        "HGCalRecHit:HGCHEBRecHits",
        "particleFlowRecHitECAL",
        "particleFlowRecHitHBHE"]

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(recHitMapProducer, hits = hits, hgcalOnly = False)
