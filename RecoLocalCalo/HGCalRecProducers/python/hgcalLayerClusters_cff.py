import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cfi import hgcalLayerClusters as hgcalLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, HGCAL_noises, hgceeDigitizer, hgchebackDigitizer, hfnoseDigitizer

hgcalLayerClusters = hgcalLayerClusters_.clone(
    timeOffset = hgceeDigitizer.tofDelay,
    plugin = dict(
        dEdXweights = dEdX.weights,
        #With the introduction of 7 regional factors (6 for silicon plus 1 for scintillator),
        #we extend fcPerMip (along with noises below) so that it is guaranteed that they have 6 entries.
        fcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP,
        thicknessCorrection = HGCalRecHit.thicknessCorrection,
        sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection,
        deltasi_index_regemfac = HGCalRecHit.deltasi_index_regemfac,
        fcPerEle = fC_per_ele,
        #Extending noises as fcPerMip, see comment above.
        noises = HGCAL_noises.values + HGCAL_noises.values,
        noiseMip = hgchebackDigitizer.digiCfg.noise
    )
)

hgcalLayerClustersHFNose = hgcalLayerClusters_.clone(
    detector = 'HFNose',
    timeOffset = hfnoseDigitizer.tofDelay,
    nHitsTime = 3,
    plugin = dict(
        dEdXweights = dEdX.weightsNose,
        maxNumberOfThickIndices = 3,
        fcPerMip = HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP,
        thicknessCorrection = HGCalRecHit.thicknessNoseCorrection,
        fcPerEle = fC_per_ele,
        noises = HGCAL_noises.values,
        noiseMip = hgchebackDigitizer.digiCfg.noise
    )
)
