import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer as _recHitMapProducer

recHitMapProducer = _recHitMapProducer.clone(
    hits = dict(HGCEE  = ("HGCalRecHit", "HGCEERecHits"),
                HGCHEF = ("HGCalRecHit", "HGCHEFRecHits"),
                HGCHEB = ("HGCalRecHit", "HGCHEBRecHits"))
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(recHitMapProducer,
                     hits = dict(HGCEE  = ("HGCalRecHit", "HGCEERecHits"),
                                 HGCHEF = ("HGCalRecHit", "HGCHEFRecHits"),
                                 HGCHEB = ("HGCalRecHit", "HGCHEBRecHits"),
                                 ECAL   = "particleFlowRecHitECAL",
                                 HBHE   = "particleFlowRecHitHBHE"),
                     hgcalOnly = False
                     )
