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
    #   doseMap = cms.string(''),
    #   sipmMap = cms.string(''),
      referenceIdark = cms.double(-1),
      referenceXtalk = cms.double(-1),
      noise_MIP = cms.double(0.01)
    ),
    use2x2 = cms.bool(True),
    type = cms.string('CLUE')
  
  ),
  detector = cms.string('all'),
  recHits = cms.InputTag('HGCalRecHit', 'HGCEERecHits'),
  timeClname = cms.string('timeLayerCluster'),
  nHitsTime = cms.uint32(3),
  mightGet = cms.optional.untracked.vstring
)
hgcalLayerClustersHSiL1Seeded = cms.EDProducer('HGCalLayerClusterProducer',
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
    #   doseMap = cms.string(''),
    #   sipmMap = cms.string(''),
      referenceIdark = cms.double(-1),
      referenceXtalk = cms.double(-1),
      noise_MIP = cms.double(0.01)
    ),
    use2x2 = cms.bool(True),
    type = cms.string('CLUE')
  
  ),
  detector = cms.string('all'),
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits'),
  timeClname = cms.string('timeLayerCluster'),
  nHitsTime = cms.uint32(3),
  mightGet = cms.optional.untracked.vstring
)
hgcalLayerClustersHSciL1Seeded = cms.EDProducer('HGCalLayerClusterProducer',
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
    #   doseMap = cms.string(''),
    #   sipmMap = cms.string(''),
      referenceIdark = cms.double(-1),
      referenceXtalk = cms.double(-1),
      noise_MIP = cms.double(0.01)
    ),
    use2x2 = cms.bool(True),
    type = cms.string('CLUE')
  
  ),
  detector = cms.string('all'),
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEBRecHits'),
  timeClname = cms.string('timeLayerCluster'),
  nHitsTime = cms.uint32(3),
  mightGet = cms.optional.untracked.vstring
)

# import FWCore.ParameterSet.Config as cms

# hgcalLayerClustersL1Seeded = cms.EDProducer("HGCalLayerClusterProducer",
#     HFNoseInput = cms.InputTag("HGCalRecHitL1Seeded","HGCHFNoseRecHits"),
#     HGCBHInput = cms.InputTag("hltRechitInRegionsHGCAL","HGCHEBRecHits"),
#     HGCEEInput = cms.InputTag("hltRechitInRegionsHGCAL","HGCEERecHits"),
#     HGCFHInput = cms.InputTag("hltRechitInRegionsHGCAL","HGCHEFRecHits"),
#     detector = cms.string('all'),
#     doSharing = cms.bool(False),
#     mightGet = cms.optional.untracked.vstring,
#     nHitsTime = cms.uint32(3),
#     plugin = cms.PSet(
#         dEdXweights = cms.vdouble(
#             0.0, 8.894541, 10.937907, 10.937907, 10.937907,
#             10.937907, 10.937907, 10.937907, 10.937907, 10.937907,
#             10.932882, 10.932882, 10.937907, 10.937907, 10.938169,
#             10.938169, 10.938169, 10.938169, 10.938169, 10.938169,
#             10.938169, 10.938169, 10.938169, 10.938169, 10.938169,
#             10.938169, 10.938169, 10.938169, 32.332097, 51.574301,
#             51.444192, 51.444192, 51.444192, 51.444192, 51.444192,
#             51.444192, 51.444192, 51.444192, 51.444192, 51.444192,
#             69.513118, 87.582044, 87.582044, 87.582044, 87.582044,
#             87.582044, 87.214571, 86.888309, 86.92952, 86.92952,
#             86.92952
#         ),
#         deltac = cms.vdouble(1.3, 1.3, 5, 0.0315),
#         deltasi_index_regemfac = cms.int32(3),
#         dependSensor = cms.bool(True),
#         ecut = cms.double(3),
#         fcPerEle = cms.double(0.00016020506),
#         fcPerMip = cms.vdouble(
#             2.06, 3.43, 5.15, 2.06, 3.43,
#             5.15
#         ),
#         kappa = cms.double(9),
#         maxNumberOfThickIndices = cms.uint32(6),
#         noiseMip = cms.PSet(
#             refToPSet_ = cms.string('HGCAL_noise_heback')
#         ),
#         noises = cms.vdouble(
#             2000.0, 2400.0, 2000.0, 2000.0, 2400.0,
#             2000.0
#         ),
#         positionDeltaRho2 = cms.double(1.69),
#         sciThicknessCorrection = cms.double(0.9),
#         thicknessCorrection = cms.vdouble(
#             0.77, 0.77, 0.77, 0.84, 0.84,
#             0.84
#         ),
#         thresholdW0 = cms.vdouble(2.9, 2.9, 2.9),
#         type = cms.string('CLUE'),
#         use2x2 = cms.bool(True),
#         verbosity = cms.untracked.uint32(3)
#     ),
#     timeClname = cms.string('timeLayerCluster'),
#     timeOffset = cms.double(5)
# )
