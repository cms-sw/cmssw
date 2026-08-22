import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

hltHgcalLayerClustersHSci = cms.EDProducer("HGCalLayerClusterProducer",
    detector = cms.string('BH'),
    mightGet = cms.optional.untracked.vstring,
    nHitsTime = cms.uint32(3),
    plugin = cms.PSet(
        dEdXweights = cms.vfloat(HGCAL_reco_constants.dEdXweights.value()),
        # Scintillator tiles use (eta, phi) coordinates, so the critical/seed/outlier
        # distances need the scintillator scale (the silicon-scale defaults are used by
        # the EE/FH instances). deltao = 0.063 reproduces the previous effective outlier
        # distance (outlierDeltaFactor = 2.0) x (scint critical distance = 0.0315).
        deltac = cms.vfloat(
            0.0315,
            0.0315,
            0.0315,
            0.0315
        ),
        deltao = cms.vfloat(
            0.063,
            0.063,
            0.063,
            0.063
        ),
        deltas = cms.vfloat(
            0.0315,
            0.0315,
            0.0315,
            0.0315
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
        type = cms.string('SciCLUE'),
        use2x2 = cms.bool(True),
        verbosity = cms.untracked.uint32(3)
    ),
    recHits = cms.InputTag("hltHGCalRecHit","HGCHEBRecHits"),
    timeClname = cms.string('timeLayerCluster')
)
