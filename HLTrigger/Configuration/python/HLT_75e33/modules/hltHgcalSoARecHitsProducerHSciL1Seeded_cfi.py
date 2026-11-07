import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants

hltHgcalSoARecHitsProducerHSciL1Seeded = cms.EDProducer("HGCalSoARecHitsProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    dEdXweights = cms.vfloat(HGCAL_reco_constants.dEdXweights.value()),
    detector = cms.string('BH'),
    ecut = cms.float(3),
    fcPerEle = cms.float(HGCAL_reco_constants.fcPerEle.value()),
    fcPerMip = cms.vfloat(HGCAL_reco_constants.fcPerMip.value()),
    maxNumberOfThickIndices = HGCAL_reco_constants.maxNumberOfThickIndices,
    noises = cms.vfloat(HGCAL_reco_constants.noises.value()),
    recHits = cms.InputTag("hltRechitInRegionsHGCAL","HGCHEBRecHits"),
    thicknessCorrection = cms.vfloat(HGCAL_reco_constants.thicknessCorrection.value()),
    noiseMip = HGCAL_reco_constants.noiseMip.noise_MIP,
    sciThicknessCorrection = HGCAL_reco_constants.sciThicknessCorrection,
)
