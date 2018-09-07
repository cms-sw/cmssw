import FWCore.ParameterSet.Config as cms

#### PF CLUSTER ECAL ####

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights, HGCalRecHit

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import fC_per_ele, hgchebackDigitizer

#energy corrector for corrected cluster producer
hgcalLayerClusters =  cms.EDProducer(
    "HGCalLayerClusterProducer",
    detector = cms.string("all"),
    doSharing = cms.bool(False),
    deltac = cms.vdouble(2.,2.,5.),
    dependSensor = cms.bool(True),
    ecut = cms.double(3.),
    kappa = cms.double(9.),
    verbosity = cms.untracked.uint32(3),
    HGCEEInput = cms.InputTag('HGCalRecHit:HGCEERecHits'),
    HGCFHInput = cms.InputTag('HGCalRecHit:HGCHEFRecHits'),
    HGCBHInput = cms.InputTag('HGCalRecHit:HGCHEBRecHits'),
    dEdXweights = cms.vdouble(dEdX_weights),
    thicknessCorrection = cms.vdouble(HGCalRecHit.thicknessCorrection),
    fcPerMip = cms.vdouble(HGCalUncalibRecHit.HGCEEConfig.fCPerMIP),
    fcPerEle = cms.double(fC_per_ele),
    noises = cms.PSet(refToPSet_ = cms.string('HGCAL_noises')),
    noiseMip = hgchebackDigitizer.digiCfg.noise_MIP
    )
