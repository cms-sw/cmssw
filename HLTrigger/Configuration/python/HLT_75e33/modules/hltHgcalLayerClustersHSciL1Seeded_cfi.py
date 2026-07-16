import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

hltHgcalLayerClustersHSciL1Seeded = cms.EDProducer("HGCalLayerClusterProducer",
    detector = cms.string('BH'),
    mightGet = cms.optional.untracked.vstring,
    nHitsTime = cms.uint32(3),
    plugin = cms.PSet(
        dEdXweights = HGCAL_reco_constants.dEdXweights,
        # Scintillator tiles use (eta, phi) coordinates, so the critical/seed/outlier
        # distances need the scintillator scale (the silicon-scale defaults are used by
        # the EE/FH instances). deltao = 0.063 reproduces the previous effective outlier
        # distance (outlierDeltaFactor = 2.0) x (scint critical distance = 0.0315).
        deltac = cms.vdouble(
            0.0315,
            0.0315,
            0.0315,
            0.0315
        ),
        deltao = cms.vdouble(
            0.063,
            0.063,
            0.063,
            0.063
        ),
        deltas = cms.vdouble(
            0.0315,
            0.0315,
            0.0315,
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
        type = cms.string('SciCLUE'),
        use2x2 = cms.bool(True),
        verbosity = cms.untracked.uint32(3)
    ),
    recHits = cms.InputTag("hltRechitInRegionsHGCAL","HGCHEBRecHits"),
    timeClname = cms.string('timeLayerCluster')
)

