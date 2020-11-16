import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cfi import hgcalLayerClusters as hgcalLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, HGCAL_noises, hgceeDigitizer, hgchebackDigitizer, hfnoseDigitizer

hgcalLayerClusters = hgcalLayerClusters_.clone(
    timeOffset = hgceeDigitizer.tofDelay,
    plugin = dict(
        dEdXweights = dEdX.weights.value(),
        #With the introduction of 7 regional factors (6 for silicon plus 1 for scintillator),
        #we extend fcPerMip (along with noises below) so that it is guaranteed that they have 6 entries.
        fcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP.value() + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessCorrection.value(),
        sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection.value(),
        deltasi_index_regemfac = HGCalRecHit.deltasi_index_regemfac.value(),
        fcPerEle = fC_per_ele,
        #Extending noises as fcPerMip, see comment above.
        noises = HGCAL_noises.values.value() + HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value()
    )
)

hgcalLayerClustersHFNose = hgcalLayerClusters_.clone(
    detector = 'HFNose',
    timeOffset = hfnoseDigitizer.tofDelay.value(),
    nHitsTime = 3,
    plugin = dict(
        dEdXweights = dEdX.weightsNose.value(),
        maxNumberOfThickIndices = 3,
        fcPerMip = HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessNoseCorrection.value(),
        fcPerEle = fC_per_ele,
        noises = HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value()
    )
)
