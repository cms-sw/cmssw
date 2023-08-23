import FWCore.ParameterSet.Config as cms

hgcalLayerClustersEEL1Seeded = cms.EDProducer('HGCalLayerClusterProducer',
  plugin = cms.PSet(
    thresholdW0 = cms.vdouble(
      2.9,
      2.9,
      2.9
    ),
    positionDeltaRho2 = cms.double(1.69),
    deltac = cms.vdouble(
      1.3,
      1.3,
      1.3,
      0.0315
    ),
    dependSensor = cms.bool(True),
    ecut = cms.double(3),
    kappa = cms.double(9),
    verbosity = cms.untracked.uint32(3),
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
            86.92952),
    thicknessCorrection = cms.vdouble(
        0.77, 0.77, 0.77, 0.84, 0.84,
             0.84
    ),
    sciThicknessCorrection = cms.double(0.9),
    deltasi_index_regemfac = cms.int32(3),
    maxNumberOfThickIndices = cms.uint32(6),
    fcPerMip = cms.vdouble(        2.06, 3.43, 5.15, 2.06, 3.43,            5.15),
    fcPerEle = cms.double(0),
    noises = cms.vdouble( 2000.0, 2400.0, 2000.0, 2000.0, 2400.0,
             2000.0),
    noiseMip = cms.PSet(
      scaleByDose = cms.bool(False),
      scaleByDoseAlgo = cms.uint32(0),
      scaleByDoseFactor = cms.double(1),
      referenceIdark = cms.double(-1),
      referenceXtalk = cms.double(-1),
      noise_MIP = cms.double(0.01)
    ),
    use2x2 = cms.bool(True),
    type = cms.string('SiCLUE')
  
  ),
  detector = cms.string('EE'),
  recHits = cms.InputTag('hltRechitInRegionsHGCAL', 'HGCEERecHits'),
  timeClname = cms.string('timeLayerCluster'),
  nHitsTime = cms.uint32(3),
  mightGet = cms.optional.untracked.vstring
)
