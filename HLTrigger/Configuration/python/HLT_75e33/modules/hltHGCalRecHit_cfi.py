import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19

hltHGCalRecHit = cms.EDProducer("HGCalRecHitProducer",
    HGCEE_cce = cms.PSet(
        refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
    ),
    HGCEE_fCPerMIP = cms.vdouble(HGCAL_reco_constants.fcPerMip[0:3]),
    HGCEE_isSiFE = cms.bool(True),
    HGCEE_keV2DIGI = cms.double(0.044259),
    HGCEE_noise_fC = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_fC')
    ),
    HGCEErechitCollection = cms.string('HGCEERecHits'),
    HGCEEuncalibRecHitCollection = cms.InputTag("hltHGCalUncalibRecHit","HGCEEUncalibRecHits"),
    HGCHEB_isSiFE = cms.bool(True),
    HGCHEB_keV2DIGI = cms.double(0.00148148148148),
    HGCHEB_noise_MIP = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_heback')
    ),
    HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
    HGCHEBuncalibRecHitCollection = cms.InputTag("hltHGCalUncalibRecHit","HGCHEBUncalibRecHits"),
    HGCHEF_cce = cms.PSet(
        refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
    ),
    HGCHEF_fCPerMIP = cms.vdouble(HGCAL_reco_constants.fcPerMip[3:6]),
    HGCHEF_isSiFE = cms.bool(True),
    HGCHEF_keV2DIGI = cms.double(0.044259),
    HGCHEF_noise_fC = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_fC')
    ),
    HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
    HGCHEFuncalibRecHitCollection = cms.InputTag("hltHGCalUncalibRecHit","HGCHEFUncalibRecHits"),
    HGCHFNose_cce = cms.PSet(
        refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
    ),
    HGCHFNose_fCPerMIP = cms.vdouble(1.25, 2.57, 3.88),
    HGCHFNose_isSiFE = cms.bool(False),
    HGCHFNose_keV2DIGI = cms.double(0.044259),
    HGCHFNose_noise_fC = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_fC')
    ),
    HGCHFNoserechitCollection = cms.string('HGCHFNoseRecHits'),
    HGCHFNoseuncalibRecHitCollection = cms.InputTag("hltHGCalUncalibRecHit","HGCHFNoseUncalibRecHits"),
    algo = cms.string('HGCalRecHitWorkerSimple'),
    constSiPar = cms.double(0.02),
    deltasi_index_regemfac = cms.int32(3),
    layerNoseWeights = cms.vdouble(
        0.0, 39.500245, 39.756638, 39.756638, 39.756638,
        39.756638, 66.020266, 92.283895, 92.283895
    ),
    layerWeights = HGCAL_reco_constants.dEdXweights,
    maxValSiPar = cms.double(10000.0),
    minValSiPar = cms.double(10.0),
    noiseSiPar = cms.double(5.5),
    rangeMask = cms.uint32(4294442496),
    rangeMatch = cms.uint32(1161838592),
    sciThicknessCorrection = HGCAL_reco_constants.sciThicknessCorrection,
    thicknessCorrection = HGCAL_reco_constants.thicknessCorrection,
    thicknessNoseCorrection = cms.vdouble(1.132, 1.092, 1.084)
)

phase2_hgcalV19.toModify(hltHGCalRecHit, 
                         HGCEE_fCPerMIP = HGCAL_reco_constants.fcPerMip[0:4],
                         HGCHEF_fCPerMIP = HGCAL_reco_constants.fcPerMip[4:8],
                         HGCHFNose_fCPerMIP = [1.25, 2.57, 3.88, 2.57],
                         )
