import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

hgcalLayerClustersHSi = cms.EDProducer("HGCalLayerClusterProducer",
    plugin = cms.PSet(
        dEdXweights = HGCAL_reco_constants.dEdXweights,
        deltac = cms.vdouble(
            1.3,
            1.3,
            1.3,
            0.0315
        ),
        deltasi_index_regemfac = cms.int32(3),
        dependSensor = cms.bool(True),
        ecut = cms.double(3),
        fcPerEle = HGCAL_reco_constants.fcPerEle,
        fcPerMip = HGCAL_reco_constants.fcPerMip,
        kappa = cms.double(9),
        maxNumberOfThickIndices = HGCAL_reco_constants.maxNumberOfThickIndices,
        noiseMip = HGCAL_reco_constants.noiseMip,
        noises = HGCAL_reco_constants.noises,
        positionDeltaRho2 = HGCAL_reco_constants.positionDeltaRho2,
        sciThicknessCorrection = HGCAL_reco_constants.sciThicknessCorrection,
        thicknessCorrection = HGCAL_reco_constants.thicknessCorrection,
        thresholdW0 = HGCAL_reco_constants.thresholdW0,
        type = cms.string('SiCLUE'),
        use2x2 = cms.bool(True),
        verbosity = cms.untracked.uint32(3)
  ),
  detector = cms.string('FH'),
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits'),
  timeClname = cms.string('timeLayerCluster'),
  nHitsTime = cms.uint32(3),
  mightGet = cms.optional.untracked.vstring
)

