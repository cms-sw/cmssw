import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

hltHgcalLayerClustersHSi = cms.EDProducer("HGCalLayerClusterProducer",
    detector = cms.string('FH'),
    mightGet = cms.optional.untracked.vstring,
    nHitsTime = cms.uint32(3),
    plugin = cms.PSet(
        dEdXweights = cms.vfloat(HGCAL_reco_constants.dEdXweights.value()),
        deltac = cms.vfloat(
            1.3,
            1.3,
            1.3,
            1.3
        ),
        deltao = cms.vfloat(
            2.6,
            2.6,
            2.6,
            2.6
        ),
        deltas = cms.vfloat(
            1.3,
            1.3,
            1.3,
            1.3
        ),
        deltasi_index_regemfac = cms.int32(3),
        dependSensor = cms.bool(True),
        ecut = cms.float(3),
        fcPerEle = cms.float(HGCAL_reco_constants.fcPerEle.value()),
        fcPerMip = cms.vfloat(HGCAL_reco_constants.fcPerMip.value()),
        kappa = cms.float(9),
        maxNumberOfThickIndices = HGCAL_reco_constants.maxNumberOfThickIndices,
        noiseMip = HGCAL_reco_constants.noiseMip,
        noises = cms.vfloat(HGCAL_reco_constants.noises.value()),
        positionDeltaRho2 = cms.float(HGCAL_reco_constants.positionDeltaRho2.value()),
        sciThicknessCorrection = cms.float(HGCAL_reco_constants.sciThicknessCorrection.value()),
        thicknessCorrection = cms.vfloat(HGCAL_reco_constants.thicknessCorrection.value()),
        thresholdW0 = cms.vfloat(HGCAL_reco_constants.thresholdW0.value()),
        type = cms.string('SiCLUE'),
        use2x2 = cms.bool(True),
        verbosity = cms.untracked.uint32(3)
    ),
    recHits = cms.InputTag("hltHGCalRecHit","HGCHEFRecHits"),
    timeClname = cms.string('timeLayerCluster')
)
