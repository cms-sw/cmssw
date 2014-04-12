import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

GamIsoEcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaRecHitExtractor'),
    DepositLabel = cms.untracked.string(''),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    isolationVariable = cms.string('et'),
    detector = cms.string('Ecal'),
    extRadius = cms.double(0.6),
    intRadius = cms.double(0.0),
    intStrip = cms.double(0.0),
    etMin = cms.double(0.0),
    energyMin = cms.double(0.095),
    subtractSuperClusterEnergy = cms.bool(False),
    tryBoth = cms.bool(True),
    vetoClustered = cms.bool(False),

    RecHitFlagToBeExcludedEB = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEB = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    RecHitFlagToBeExcludedEE = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEE = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded
    
#    severityLevelCut = cms.int32(4),
#     severityRecHitThreshold = cms.double(5.0),
#     spikeIdString = cms.string('kSwissCrossBordersIncluded'),
#     spikeIdThreshold = cms.double(0.95),

#    recHitFlagsToBeExcluded = cms.vstring(
#        'kFaultyHardware',
#        'kPoorCalib',
#        ecalRecHitFlag_kSaturated,
#        ecalRecHitFlag_kLeadingEdgeRecovered,
#        ecalRecHitFlag_kNeighboursRecovered,
#        'kTowerRecovered',
#        'kDead'
#    ),
)

