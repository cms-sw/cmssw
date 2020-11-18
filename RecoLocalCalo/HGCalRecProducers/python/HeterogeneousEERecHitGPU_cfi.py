import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

EERecHitGPUProd = cms.EDProducer('EERecHitGPU',
                                 HGCEEUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCEEUncalibRecHits'),
                                 HGCEE_keV2DIGI = HGCalRecHit.__dict__['HGCEE_keV2DIGI'],
                                 minValSiPar    = HGCalRecHit.__dict__['minValSiPar'],
                                 maxValSiPar    = HGCalRecHit.__dict__['maxValSiPar'],
                                 constSiPar     = HGCalRecHit.__dict__['constSiPar'],
                                 noiseSiPar     = HGCalRecHit.__dict__['noiseSiPar'],
                                 HGCEE_fCPerMIP = HGCalRecHit.__dict__['HGCEE_fCPerMIP'],
                                 HGCEE_isSiFE   = HGCalRecHit.__dict__['HGCEE_isSiFE'],
                                 HGCEE_noise_fC = HGCalRecHit.__dict__['HGCEE_noise_fC'],
                                 HGCEE_cce      = HGCalRecHit.__dict__['HGCEE_cce'],
                                 rcorr          = cms.vdouble( HGCalRecHit.__dict__['thicknessCorrection'][0:3] ),
                                 weights        = HGCalRecHit.__dict__['layerWeights'] )
