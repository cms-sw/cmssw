import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cfi import hgcalLayerClusters as hgcalLayerClusters_
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters as hgcalMergeLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, HGCAL_noises, hgceeDigitizer, hgchebackDigitizer, hfnoseDigitizer

hgcalLayerClustersEE = hgcalLayerClusters_.clone(
    detector = 'EE',
    recHits = "HGCalRecHit:HGCEERecHits",
    plugin = dict(
        dEdXweights = HGCalRecHit.layerWeights.value(),
        #With the introduction of 7 regional factors (6 for silicon plus 1 for scintillator),
        #we extend fcPerMip (along with noises below) so that it is guaranteed that they have 6 entries.
        fcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP.value() + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessCorrection.value(),
        sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection.value(),
        deltasi_index_regemfac = HGCalRecHit.deltasi_index_regemfac.value(),
        fcPerEle = fC_per_ele,
        #Extending noises as fcPerMip, see comment above.
        noises = HGCAL_noises.values.value() + HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value(),
        type = "SiCLUE"
    )
)

hgcalLayerClustersHSi = hgcalLayerClusters_.clone(
    detector = 'FH',
    recHits = "HGCalRecHit:HGCHEFRecHits",
    plugin = dict(
        dEdXweights = HGCalRecHit.layerWeights.value(),
        #With the introduction of 7 regional factors (6 for silicon plus 1 for scintillator),
        #we extend fcPerMip (along with noises below) so that it is guaranteed that they have 6 entries.
        fcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP.value() + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessCorrection.value(),
        sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection.value(),
        deltasi_index_regemfac = HGCalRecHit.deltasi_index_regemfac.value(),
        fcPerEle = fC_per_ele,
        #Extending noises as fcPerMip, see comment above.
        noises = HGCAL_noises.values.value() + HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value(),
        type = "SiCLUE"
    )
)

hgcalLayerClustersHSci = hgcalLayerClusters_.clone(
    detector = 'BH',
    recHits = "HGCalRecHit:HGCHEBRecHits",
    plugin = dict(
        dEdXweights = HGCalRecHit.layerWeights.value(),
        #With the introduction of 7 regional factors (6 for silicon plus 1 for scintillator),
        #we extend fcPerMip (along with noises below) so that it is guaranteed that they have 6 entries.
        fcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP.value() + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessCorrection.value(),
        sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection.value(),
        deltasi_index_regemfac = HGCalRecHit.deltasi_index_regemfac.value(),
        fcPerEle = fC_per_ele,
        #Extending noises as fcPerMip, see comment above.
        noises = HGCAL_noises.values.value() + HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value(),
        type = "SciCLUE"
    )
)

hgcalLayerClustersHFNose = hgcalLayerClusters_.clone(
    detector = 'HFNose',
    recHits = "HGCalRecHit:HGCHFNoseRecHits",
    nHitsTime = 3,
    plugin = dict(
        dEdXweights = HGCalRecHit.layerNoseWeights.value(),
        maxNumberOfThickIndices = 3,
        fcPerMip = HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP.value(),
        thicknessCorrection = HGCalRecHit.thicknessNoseCorrection.value(),
        fcPerEle = fC_per_ele,
        noises = HGCAL_noises.values.value(),
        noiseMip = hgchebackDigitizer.digiCfg.noise.value(),
        type = "SciCLUE"
    )
)

from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import fCPerMIP_mean_V19
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import nonAgedNoises_v9_v19

#The v19 geometry adds a fourth silicon sensor category (HD 200um, type 3), so
#the silicon constants have 4 entries per section (8 regional factors plus 1
#for scintillator) and the CE-H offset moves from 3 to 4. The clones above copy
#the pre-modifier defaults at import time, so the v19 values must be set here
#explicitly.
_v19SiPlugin = dict(
    deltasi_index_regemfac = 4,
    maxNumberOfThickIndices = 8,
    thicknessCorrection = [0.75, 0.76, 0.75, 0.76, 0.85, 0.85, 0.84, 0.85],
    fcPerMip = fCPerMIP_mean_V19.value() + fCPerMIP_mean_V19.value(),
    noises = nonAgedNoises_v9_v19 + nonAgedNoises_v9_v19,
)
for _clusters in (hgcalLayerClustersEE, hgcalLayerClustersHSi, hgcalLayerClustersHSci):
    phase2_hgcalV19.toModify(_clusters, plugin = dict(**_v19SiPlugin))

hgcalMergeLayerClusters = hgcalMergeLayerClusters_.clone(
)

layerClusters = cms.VInputTag('hgcalLayerClustersEE', 'hgcalLayerClustersHSi', 'hgcalLayerClustersHSci', 'barrelLayerClustersEB', 'barrelLayerClustersHB')
time_layerClusters = cms.VInputTag('hgcalLayerClustersEE:timeLayerCluster', 'hgcalLayerClustersHSi:timeLayerCluster', 'hgcalLayerClustersHSci:timeLayerCluster', 'barrelLayerClustersEB:timeLayerCluster', 'barrelLayerClustersHB:timeLayerCluster')
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(hgcalMergeLayerClusters, layerClusters = layerClusters, time_layerclusters = time_layerClusters)
