import FWCore.ParameterSet.Config as cms
from ..psets.hgcal_reco_constants_cfi import HGCAL_reco_constants as HGCAL_reco_constants
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsProducerHSci = cms.EDProducer("HGCalSoARecHitsProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    dEdXweights = HGCAL_reco_constants.dEdXweights,
    detector = cms.string('BH'),
    ecut = cms.double(3),
    fcPerEle = HGCAL_reco_constants.fcPerEle,
    fcPerMip = HGCAL_reco_constants.fcPerMip,
    maxNumberOfThickIndices = HGCAL_reco_constants.maxNumberOfThickIndices,
    noises = HGCAL_reco_constants.noises,
    recHits = cms.InputTag("hltHGCalRecHit","HGCHEBRecHits"),
    thicknessCorrection = HGCAL_reco_constants.thicknessCorrection,
    noiseMip = HGCAL_reco_constants.noiseMip.noise_MIP,
    sciThicknessCorrection = HGCAL_reco_constants.sciThicknessCorrection,
)

hltHgcalSoARecHitsProducerHSciSerialSync = makeSerialClone(hltHgcalSoARecHitsProducerHSci)
