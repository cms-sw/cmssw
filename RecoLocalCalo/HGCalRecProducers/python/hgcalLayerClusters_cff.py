import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters as hgcalMergeLayerClusters_

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, HGCAL_noises, HGCAL_noise_heback, hgceeDigitizer, hgchebackDigitizer, hfnoseDigitizer

from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import fCPerMIP_mean_V19
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import nonAgedNoises_v9_v19


#####################################################################
# CLUEstering layer-clustering chain 
#
# The chain per detector is:
#   hgcalSoARecHits<Det>       (HGCRecHit  -> HGCalSoARecHits SoA)
#   hgcalCLUEstering<Det>      (CLUE clustering on the SoA)
#   hgcalSoALayerClusters<Det> (build the CaloCluster SoA)
#   _fromSoA<Det>              (SoA -> legacy reco::CaloCluster products)
#####################################################################
from RecoLocalCalo.HGCalRecProducers.hgCalSoARecHitsProducer_cfi import hgCalSoARecHitsProducer
from RecoLocalCalo.HGCalRecProducers.hgCalCLUEsteringLayerClustersProducer_cfi import hgCalCLUEsteringLayerClustersProducer
from RecoLocalCalo.HGCalRecProducers.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
from RecoLocalCalo.HGCalRecProducers.hgCalLayerClustersFromSoAProducer_cfi import hgCalLayerClustersFromSoAProducer

# EE/FH/BH share the same silicon energy-threshold constants. Scintillator
# (BH) also uses them: its own thickness index is out of range and the
# producer then uses a zero threshold, exactly as in the CPU algorithm. Its
# noise constants, however, are scintillator-specific -- see below.
_siFcPerMip = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP.value() + HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP.value()
_siNoises = HGCAL_noises.values.value() + HGCAL_noises.values.value()
_siThicknessCorrection = HGCalRecHit.thicknessCorrection.value()
_siDEdXweights = HGCalRecHit.layerWeights.value()

def _makeSoARecHits(det, recHits, **kwargs):
    return hgCalSoARecHitsProducer.clone(
        detector = det,
        recHits = recHits,
        maxNumberOfThickIndices = 6,
        fcPerMip = _siFcPerMip,
        thicknessCorrection = _siThicknessCorrection,
        noises = _siNoises,
        dEdXweights = _siDEdXweights,
        **kwargs
    )

# ---- EE (silicon, electromagnetic) ----
hgcalSoARecHitsEE = _makeSoARecHits('EE', "HGCalRecHit:HGCEERecHits")
hgcalCLUEsteringEE = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsEE', detector = 'EE',
    deltac = 1.3, kappa = 9., outlierDeltaFactor = 2.)
hgcalSoALayerClustersEE = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsEE', hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringEE',
    detector = 'EE')
_fromSoAEE = hgCalLayerClustersFromSoAProducer.clone(
    src = 'hgcalSoALayerClustersEE',
    hgcalRecHitsSoA = 'hgcalSoARecHitsEE',
    hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringEE',
    detector = 'EE')

# ---- HSi / FH (silicon, hadronic) ----
hgcalSoARecHitsHSi = _makeSoARecHits('FH', "HGCalRecHit:HGCHEFRecHits")
hgcalCLUEsteringHSi = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSi', detector = 'FH',
    deltac = 1.3, kappa = 9., outlierDeltaFactor = 2.)
hgcalSoALayerClustersHSi = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSi', hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHSi',
    detector = 'FH')
_fromSoAFH = hgCalLayerClustersFromSoAProducer.clone(
    src = 'hgcalSoALayerClustersHSi',
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSi',
    hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHSi',
    detector = 'FH')

# ---- HSci / BH (scintillator, hadronic) ----
hgcalSoARecHitsHSci = _makeSoARecHits(
    'BH', "HGCalRecHit:HGCHEBRecHits",
    noiseMip = HGCAL_noise_heback.noise_MIP.value(),
    sciThicknessCorrection = HGCalRecHit.sciThicknessCorrection.value())
hgcalCLUEsteringHSci = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSci', detector = 'BH',
    deltac = 0.0315, kappa = 9., outlierDeltaFactor = 2.)
hgcalSoALayerClustersHSci = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSci', hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHSci',
    detector = 'BH')
_fromSoABH = hgCalLayerClustersFromSoAProducer.clone(
    src = 'hgcalSoALayerClustersHSci',
    hgcalRecHitsSoA = 'hgcalSoARecHitsHSci',
    hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHSci',
    detector = 'BH')

# ---- HFNose (silicon, only under the phase2_hfnose modifier) ----
hgcalSoARecHitsHFNose = hgCalSoARecHitsProducer.clone(
    detector = 'HFNose',
    recHits = "HGCalRecHit:HGCHFNoseRecHits",
    maxNumberOfThickIndices = 3,
    fcPerMip = HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP.value(),
    thicknessCorrection = HGCalRecHit.thicknessNoseCorrection.value(),
    noises = HGCAL_noises.values.value(),
    dEdXweights = HGCalRecHit.layerNoseWeights.value())
hgcalCLUEsteringHFNose = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHFNose', detector = 'HFNose',
    deltac = 1.3, kappa = 9., outlierDeltaFactor = 2.)
hgcalSoALayerClustersHFNose = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = 'hgcalSoARecHitsHFNose', hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHFNose',
    detector = 'HFNose')
_fromSoAHFNose = hgCalLayerClustersFromSoAProducer.clone(
    src = 'hgcalSoALayerClustersHFNose',
    hgcalRecHitsSoA = 'hgcalSoARecHitsHFNose',
    hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringHFNose',
    detector = 'HFNose',
    nHitsTime = 3)

_v19SiSoAParams = dict(
    maxNumberOfThickIndices = 8,
    thicknessCorrection = [0.75, 0.76, 0.75, 0.76, 0.85, 0.85, 0.84, 0.85],
    fcPerMip = fCPerMIP_mean_V19.value() + fCPerMIP_mean_V19.value(),
    noises = nonAgedNoises_v9_v19 + nonAgedNoises_v9_v19,
)
for _soa in (hgcalSoARecHitsEE, hgcalSoARecHitsHSi, hgcalSoARecHitsHSci):
    phase2_hgcalV19.toModify(_soa, **_v19SiSoAParams)

hgcalLayerClustersAlpakaTask = cms.Task(
    hgcalSoARecHitsEE,   hgcalCLUEsteringEE,   hgcalSoALayerClustersEE,
    hgcalSoARecHitsHSi,  hgcalCLUEsteringHSi,  hgcalSoALayerClustersHSi,
    hgcalSoARecHitsHSci, hgcalCLUEsteringHSci, hgcalSoALayerClustersHSci,
)
hgcalLayerClustersHFNoseAlpakaTask = cms.Task(
    hgcalSoARecHitsHFNose, hgcalCLUEsteringHFNose, hgcalSoALayerClustersHFNose,
)

hgcalLayerClustersEE     = _fromSoAEE
hgcalLayerClustersHSi    = _fromSoAFH
hgcalLayerClustersHSci   = _fromSoABH
hgcalLayerClustersHFNose = _fromSoAHFNose

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
