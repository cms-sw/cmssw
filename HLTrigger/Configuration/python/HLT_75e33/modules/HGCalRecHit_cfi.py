import FWCore.ParameterSet.Config as cms

HGCalRecHit = cms.EDProducer("HGCalRecHitProducer",
    HGCEE_cce = cms.PSet(
        refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
    ),
    HGCEE_fCPerMIP = cms.vdouble(2.06, 3.43, 5.15),
    HGCEE_isSiFE = cms.bool(True),
    HGCEE_keV2DIGI = cms.double(0.044259),
    HGCEE_noise_fC = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_fC')
    ),
    HGCEErechitCollection = cms.string('HGCEERecHits'),
    HGCEEuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCEEUncalibRecHits"),
    HGCHEB_isSiFE = cms.bool(True),
    HGCHEB_keV2DIGI = cms.double(0.00148148148148),
    HGCHEB_noise_MIP = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_heback')
    ),
    HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
    HGCHEBuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEBUncalibRecHits"),
    HGCHEF_cce = cms.PSet(
        refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
    ),
    HGCHEF_fCPerMIP = cms.vdouble(2.06, 3.43, 5.15),
    HGCHEF_isSiFE = cms.bool(True),
    HGCHEF_keV2DIGI = cms.double(0.044259),
    HGCHEF_noise_fC = cms.PSet(
        refToPSet_ = cms.string('HGCAL_noise_fC')
    ),
    HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
    HGCHEFuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEFUncalibRecHits"),
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
    HGCHFNoseuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHFNoseUncalibRecHits"),
    algo = cms.string('HGCalRecHitWorkerSimple'),
    constSiPar = cms.double(0.02),
    deltasi_index_regemfac = cms.int32(3),
    layerNoseWeights = cms.vdouble(
        0.0, 39.500245, 39.756638, 39.756638, 39.756638,
        39.756638, 66.020266, 92.283895, 92.283895
    ),
    layerWeights = cms.vdouble(
        0.0, 8.894541, 10.937907, 10.937907, 10.937907,
        10.937907, 10.937907, 10.937907, 10.937907, 10.937907,
        10.932882, 10.932882, 10.937907, 10.937907, 10.938169,
        10.938169, 10.938169, 10.938169, 10.938169, 10.938169,
        10.938169, 10.938169, 10.938169, 10.938169, 10.938169,
        10.938169, 10.938169, 10.938169, 32.332097, 51.574301,
        51.444192, 51.444192, 51.444192, 51.444192, 51.444192,
        51.444192, 51.444192, 51.444192, 51.444192, 51.444192,
        69.513118, 87.582044, 87.582044, 87.582044, 87.582044,
        87.582044, 87.214571, 86.888309, 86.92952, 86.92952,
        86.92952
    ),
    maxValSiPar = cms.double(10000.0),
    minValSiPar = cms.double(10.0),
    noiseSiPar = cms.double(5.5),
    rangeMask = cms.uint32(4294442496),
    rangeMatch = cms.uint32(1161838592),
    sciThicknessCorrection = cms.double(0.9),
    thicknessCorrection = cms.vdouble(
        0.77, 0.77, 0.77, 0.84, 0.84,
        0.84
    ),
    thicknessNoseCorrection = cms.vdouble(1.132, 1.092, 1.084)
)
