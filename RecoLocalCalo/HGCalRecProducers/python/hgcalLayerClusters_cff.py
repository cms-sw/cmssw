import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cfi import hgcalLayerClusters as hgcalLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, hgceeDigitizer, hgchebackDigitizer

hgcalLayerClusters = hgcalLayerClusters_.clone()

hgcalLayerClusters.dEdXweights = cms.vdouble(dEdX.weights)
hgcalLayerClusters.fcPerMip = cms.vdouble(HGCalUncalibRecHit.HGCEEConfig.fCPerMIP)
hgcalLayerClusters.thicknessCorrection = cms.vdouble(HGCalRecHit.thicknessCorrection)
hgcalLayerClusters.fcPerEle = cms.double(fC_per_ele)
hgcalLayerClusters.timeOffset = hgceeDigitizer.tofDelay
hgcalLayerClusters.noises = cms.PSet(refToPSet_ = cms.string('HGCAL_noises'))
hgcalLayerClusters.noiseMip = hgchebackDigitizer.digiCfg.noise_MIP
