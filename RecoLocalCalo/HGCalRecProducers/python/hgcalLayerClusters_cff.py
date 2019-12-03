import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cfi import hgcalLayerClusters as hgcalLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, hgceeDigitizer, hgchebackDigitizer, hfnoseDigitizer

hgcalLayerClusters = hgcalLayerClusters_.clone()

hgcalLayerClusters.timeOffset = hgceeDigitizer.tofDelay
hgcalLayerClusters.plugin.dEdXweights = cms.vdouble(dEdX.weights)
hgcalLayerClusters.plugin.fcPerMip = cms.vdouble(HGCalUncalibRecHit.HGCEEConfig.fCPerMIP)
hgcalLayerClusters.plugin.thicknessCorrection = cms.vdouble(HGCalRecHit.thicknessCorrection)
hgcalLayerClusters.plugin.fcPerEle = cms.double(fC_per_ele)
hgcalLayerClusters.plugin.noises = cms.PSet(refToPSet_ = cms.string('HGCAL_noises'))
hgcalLayerClusters.plugin.noiseMip = hgchebackDigitizer.digiCfg.noise


hgcalLayerClustersHFNose = hgcalLayerClusters_.clone(
    detector = cms.string('HFNose'),
    timeOffset = hfnoseDigitizer.tofDelay,
    plugin = cms.PSet(
        dEdXweights = cms.vdouble(dEdX.weightsNose),
        fcPerMip = cms.vdouble(HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP),
        thicknessCorrection = cms.vdouble(HGCalRecHit.thicknessNoseCorrection),
        fcPerEle = cms.double(fC_per_ele),
        noises = cms.PSet(refToPSet_ = cms.string('HGCAL_noises')),
        noiseMip = hgchebackDigitizer.digiCfg.noise
    )
)
