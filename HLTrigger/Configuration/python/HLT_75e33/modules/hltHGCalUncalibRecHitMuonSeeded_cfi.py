import FWCore.ParameterSet.Config as cms
from ..psets.HGCAL_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19


hltHGCalUncalibRecHitMuonSeeded = cms.EDProducer("HGCalUncalibRecHitProducer",
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
    HGCEEdigiCollection = cms.InputTag("hltHgcalDigisMuonSeeded","EE"),
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
    HGCHEBdigiCollection = cms.InputTag("hltHgcalDigisMuonSeeded","HEback"),
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
    HGCHEFdigiCollection = cms.InputTag("hltHgcalDigisMuonSeeded","HEfront"),
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
    computeLocalTime = cms.bool(True),
    algo = cms.string('HGCalUncalibRecHitWorkerWeights')
)



_modifiedHGCEEConfig_v19 = hltHGCalUncalibRecHitMuonSeeded.HGCEEConfig.clone(
    fCPerMIP = HGCAL_reco_constants.fcPerMip[0:4]
)
_modifiedHGCHEFConfig_v19 = hltHGCalUncalibRecHitMuonSeeded.HGCHEFConfig.clone(
    fCPerMIP = HGCAL_reco_constants.fcPerMip[4:8]
)
_modifiedHGCHFNoseConfig_v19 = hltHGCalUncalibRecHitMuonSeeded.HGCHFNoseConfig.clone(
    fCPerMIP = [1.25, 2.57, 3.88, 2.57]
)

phase2_hgcalV19.toModify(
    hltHGCalUncalibRecHitMuonSeeded,
    HGCEEConfig = _modifiedHGCEEConfig_v19,
    HGCHEFConfig = _modifiedHGCHEFConfig_v19,
    HGCHFNoseConfig = _modifiedHGCHFNoseConfig_v19
)
