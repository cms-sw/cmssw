import FWCore.ParameterSet.Config as cms

hgcalLayerClustersHSci = cms.EDProducer("HGCalLayerClusterProducer",
    detector = cms.string('BH'),
    mightGet = cms.optional.untracked.vstring,
    nHitsTime = cms.uint32(3),
    plugin = cms.PSet(
        dEdXweights = cms.vdouble(
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
        deltac = cms.vdouble(1.3, 1.3, 1.3, 0.0315),
        deltasi_index_regemfac = cms.int32(3),
        dependSensor = cms.bool(True),
        ecut = cms.double(3),
        fcPerEle = cms.double(0),
        fcPerMip = cms.vdouble(
            2.06, 3.43, 5.15, 2.06, 3.43,
            5.15
        ),
        kappa = cms.double(9),
        maxNumberOfThickIndices = cms.uint32(6),
        noiseMip = cms.PSet(
            noise_MIP = cms.double(0.01),
            referenceIdark = cms.double(-1),
            referenceXtalk = cms.double(-1),
            scaleByDose = cms.bool(False),
            scaleByDoseAlgo = cms.uint32(0),
            scaleByDoseFactor = cms.double(1)
        ),
        noises = cms.vdouble(
            2000.0, 2400.0, 2000.0, 2000.0, 2400.0,
            2000.0
        ),
        positionDeltaRho2 = cms.double(1.69),
        sciThicknessCorrection = cms.double(0.9),
        thicknessCorrection = cms.vdouble(
            0.77, 0.77, 0.77, 0.84, 0.84,
            0.84
        ),
        thresholdW0 = cms.vdouble(2.9, 2.9, 2.9),
        type = cms.string('SciCLUE'),
        use2x2 = cms.bool(True),
        verbosity = cms.untracked.uint32(3)
    ),
    recHits = cms.InputTag("HGCalRecHit","HGCHEBRecHits"),
    timeClname = cms.string('timeLayerCluster')
)
