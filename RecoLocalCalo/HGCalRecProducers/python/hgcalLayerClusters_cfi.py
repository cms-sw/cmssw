import FWCore.ParameterSet.Config as cms

#### PF CLUSTER ECAL ####

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, nonAgedNoises, hgchebackDigitizer

#energy corrector for corrected cluster producer
hgcalLayerClusters =  cms.EDProducer(
    "HGCalClusterTestProducer",
    detector = cms.string("all"),
    doSharing = cms.bool(False),
    deltac = cms.vdouble(2.,2.,2.),
    ecut = cms.double(0.01),
    kappa = cms.double(10.),
    multiclusterRadii = cms.vdouble(0.015,0.015,0.015),
    realSpaceCone = cms.bool(False),
    minClusters = cms.uint32(3),
    verbosity = cms.untracked.uint32(3),
    HGCEEInput = cms.InputTag('HGCalRecHit:HGCEERecHits'),
    HGCFHInput = cms.InputTag('HGCalRecHit:HGCHEFRecHits'),
    HGCBHInput = cms.InputTag('HGCalRecHit:HGCHEBRecHits'),
    dependSensor = cms.bool(False),
    dEdXweights = cms.vdouble(dEdX_weights),
    thicknessCorrection = cms.vdouble(HGCalRecHit.thicknessCorrection),
    fcPerMip = cms.vdouble(HGCalUncalibRecHit.HGCEEConfig.fCPerMIP),
    fcPerEle = cms.double(fC_per_ele),
    nonAgedNoises = cms.vdouble(nonAgedNoises),
    noiseMip = hgchebackDigitizer.digiCfg.noise_MIP
    )

