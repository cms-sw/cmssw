import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLCommonAlgos.ftlSimpleRecHitAlgo_cff import ftlSimpleRecHitAlgo

_barrelAlgo = ftlSimpleRecHitAlgo.clone()
_endcapAlgo = ftlSimpleRecHitAlgo.clone()

ftlRecHits = cms.EDProducer(
    "FTLRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelUncalibratedRecHits = cms.InputTag('ftlUncalibratedRecHits:FTLBarrel'),
    endcapUncalibratedRecHits = cms.InputTag('ftlUncalibratedRecHits:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap')
)
