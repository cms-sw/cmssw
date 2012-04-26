import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

interestingGamIsoDetId = cms.EDProducer("GamIsoDetIdCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    emObjectLabel = cms.InputTag("photons"),
    etCandCut = cms.double(0.0),
    energyCut = cms.double(0.040),
    etCut = cms.double(0),
    outerRadius = cms.double(0.6),
    innerRadius = cms.double(0.0),
    interestingDetIdCollection = cms.string(''),

    RecHitFlagToBeExcludedEB = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEB = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    RecHitFlagToBeExcludedEE = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEE = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
)
