import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

HEBRecHitGPUProd = cms.EDProducer('HEBRecHitGPU',
                                  HGCHEBUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEBUncalibRecHits'),
                                  HGCHEB_keV2DIGI  = HGCalRecHit.__dict__['HGCHEB_keV2DIGI'],
                                  HGCHEB_noise_MIP = HGCalRecHit.__dict__['HGCHEB_noise_MIP'],
                                  weights          = HGCalRecHit.__dict__['layerWeights'] )
