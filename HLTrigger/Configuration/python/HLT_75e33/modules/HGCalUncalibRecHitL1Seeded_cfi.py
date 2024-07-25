import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

HGCalUncalibRecHitL1Seeded = cms.EDProducer("HGCalUncalibRecHitProducer",
    HGCEEConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(HGCAL_reco_constants.fcPerMip[0:3]),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        toaLSB_ns = cms.double(0.0244),
        tofDelay = cms.double(-9)
    ),
    HGCEEdigiCollection = cms.InputTag("hgcalDigisL1Seeded","EE"),
    HGCEEhitCollection = cms.string('HGCEEUncalibRecHits'),
    HGCHEBConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(68.75),
        fCPerMIP = cms.vdouble(1.0, 1.0, 1.0),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(55),
        tdcSaturation = cms.double(1000),
        toaLSB_ns = cms.double(0.0244),
        tofDelay = cms.double(-14)
    ),
    HGCHEBdigiCollection = cms.InputTag("hgcalDigisL1Seeded","HEback"),
    HGCHEBhitCollection = cms.string('HGCHEBUncalibRecHits'),
    HGCHEFConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(HGCAL_reco_constants.fcPerMip[3:6]),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        toaLSB_ns = cms.double(0.0244),
        tofDelay = cms.double(-11)
    ),
    HGCHEFdigiCollection = cms.InputTag("hgcalDigisL1Seeded","HEfront"),
    HGCHEFhitCollection = cms.string('HGCHEFUncalibRecHits'),
    HGCHFNoseConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(1.25, 2.57, 3.88),
        isSiFE = cms.bool(False),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        toaLSB_ns = cms.double(0.0244),
        tofDelay = cms.double(-33)
    ),
    HGCHFNosedigiCollection = cms.InputTag("hfnoseDigis","HFNose"),
    HGCHFNosehitCollection = cms.string('HGCHFNoseUncalibRecHits'),
    computeLocalTime = cms.bool(False),
    algo = cms.string('HGCalUncalibRecHitWorkerWeights')
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(HGCalUncalibRecHitL1Seeded, computeLocalTime = cms.bool(True))
