import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

HEFRecHitGPUProd = cms.EDProducer('HEFRecHitGPU',
                                  HGCHEFUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit','HGCHEFUncalibRecHits'),
                                  HGCHEF_keV2DIGI  = HGCalRecHit.__dict__['HGCHEF_keV2DIGI'],
                                  minValSiPar     = HGCalRecHit.__dict__['minValSiPar'],
                                  maxValSiPar     = HGCalRecHit.__dict__['maxValSiPar'],
                                  constSiPar      = HGCalRecHit.__dict__['constSiPar'],
                                  noiseSiPar      = HGCalRecHit.__dict__['noiseSiPar'],
                                  HGCHEF_fCPerMIP = HGCalRecHit.__dict__['HGCHEF_fCPerMIP'],
                                  HGCHEF_isSiFE   = HGCalRecHit.__dict__['HGCHEF_isSiFE'],
                                  HGCHEF_noise_fC = HGCalRecHit.__dict__['HGCHEF_noise_fC'],
                                  HGCHEF_cce      = HGCalRecHit.__dict__['HGCHEF_cce'],
                                  rcorr           = cms.vdouble( HGCalRecHit.__dict__['thicknessCorrection'][3:6] ),
                                  weights         = HGCalRecHit.__dict__['layerWeights'] )
