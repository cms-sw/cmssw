import FWCore.ParameterSet.Config as cms

interestingEcalDetIdEB = cms.EDProducer("InterestingDetIdCollectionProducer",
    basicClustersLabel = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    etaSize = cms.int32(5),
    interestingDetIdCollection = cms.string(''),
    phiSize = cms.int32(5)
)

interestingEcalDetIdEE = cms.EDProducer("InterestingDetIdCollectionProducer",
    basicClustersLabel = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    etaSize = cms.int32(5),
    interestingDetIdCollection = cms.string(''),
    phiSize = cms.int32(5)
)

# rechits associated to high pt tracks for HSCP

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

interestingTrackEcalDetIds = cms.EDProducer('InterestingTrackEcalDetIdProducer',
    TrackAssociatorParameterBlock,
    TrackCollection = cms.InputTag("generalTracks"),
    MinTrackPt      = cms.double(50.0)
)


reducedEcalRecHitsEB = cms.EDProducer("ReducedRecHitCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    interestingDetIdCollections = cms.VInputTag(
            # ecal
            cms.InputTag("interestingEcalDetIdEB"),
            # egamma
            cms.InputTag("interestingEleIsoDetIdEB"),
            cms.InputTag("interestingGamIsoDetIdEB"),
            # tau
            #cms.InputTag("caloRecoTauProducer"),
            #pf
            cms.InputTag("pfElectronInterestingEcalDetIdEB"),
            # muons
            cms.InputTag("muonEcalDetIds"),
            # high pt tracks
            cms.InputTag("interestingTrackEcalDetIds")
            ),
    reducedHitsCollection = cms.string('')
)

reducedEcalRecHitsEE = cms.EDProducer("ReducedRecHitCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    interestingDetIdCollections = cms.VInputTag(
            # ecal
            cms.InputTag("interestingEcalDetIdEE"),
            # egamma
            cms.InputTag("interestingEleIsoDetIdEE"),
            cms.InputTag("interestingGamIsoDetIdEE"),
            # tau
            #cms.InputTag("caloRecoTauProducer"),
            #pf
            cms.InputTag("pfElectronInterestingEcalDetIdEE"),
            # muons
            cms.InputTag("muonEcalDetIds"),
            # high pt tracks
            cms.InputTag("interestingTrackEcalDetIds")
            ),
    reducedHitsCollection = cms.string('')
)


#selected digis
from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cff import *

reducedEcalRecHitsSequence = cms.Sequence(interestingEcalDetIdEB*interestingEcalDetIdEE*interestingTrackEcalDetIds*reducedEcalRecHitsEB*reducedEcalRecHitsEE*seldigis)
