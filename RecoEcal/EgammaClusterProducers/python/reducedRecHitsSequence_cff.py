import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi

interestingEcalDetIdEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

interestingEcalDetIdEBU = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridBarrelBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

interestingEcalDetIdEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )

interestingEcalDetIdPFEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterECAL","particleFlowBasicClusterECALBarrel"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

interestingEcalDetIdPFEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterECAL","particleFlowBasicClusterECALEndcap"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )

interestingEcalDetIdPFES = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterECAL","particleFlowBasicClusterECALPreshower"),
    recHitsLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    severityLevel = cms.int32(-1),
    keepNextToDead = cms.bool(False),
    keepNextToBoundary = cms.bool(False)    
    )

interestingEcalDetIdRefinedEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowEGamma","EBEEClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

interestingEcalDetIdRefinedEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowEGamma","EBEEClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )

interestingEcalDetIdRefinedES = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowEGamma","ESClusters"),
    recHitsLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    severityLevel = cms.int32(-1),
    keepNextToDead = cms.bool(False),
    keepNextToBoundary = cms.bool(False)    
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
            cms.InputTag("interestingEcalDetIdEBU"),
            #ged
            cms.InputTag("interestingEcalDetIdPFEB"),
            cms.InputTag("interestingEcalDetIdRefinedEB"),
            # egamma
            cms.InputTag("interestingGedEleIsoDetIdEB"),
            cms.InputTag("interestingGedGamIsoDetIdEB"),
            cms.InputTag("interestingGamIsoDetIdEB"),
            # tau
            #cms.InputTag("caloRecoTauProducer"),
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
            #ged
            cms.InputTag("interestingEcalDetIdPFEE"),
            cms.InputTag("interestingEcalDetIdRefinedEE"),            
            # egamma
            cms.InputTag("interestingGedEleIsoDetIdEE"),
            cms.InputTag("interestingGedGamIsoDetIdEE"),
            cms.InputTag("interestingGamIsoDetIdEE"),
            # tau
            #cms.InputTag("caloRecoTauProducer"),
            # muons
            cms.InputTag("muonEcalDetIds"),
            # high pt tracks
            cms.InputTag("interestingTrackEcalDetIds")
            ),
    reducedHitsCollection = cms.string('')
)

reducedEcalRecHitsES = cms.EDProducer("ReducedESRecHitCollectionProducer",
                                      scEtThreshold = cms.double(15),
                                      EcalRecHitCollectionES = cms.InputTag('ecalPreshowerRecHit','EcalRecHitsES'),
                                      EndcapSuperClusterCollection = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),
                                      OutputLabel_ES = cms.string(''),
                                      interestingDetIds = cms.VInputTag(
                                        cms.InputTag("interestingEcalDetIdPFES"),
                                        cms.InputTag("interestingEcalDetIdRefinedES"), 
                                      )
)

#selected digis
from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cff import *

reducedEcalRecHitsSequence = cms.Sequence(interestingEcalDetIdEB*interestingEcalDetIdEBU*
                                          interestingEcalDetIdEE*
                                          interestingEcalDetIdPFEB*interestingEcalDetIdPFEE*interestingEcalDetIdPFES*
                                          interestingEcalDetIdRefinedEB*interestingEcalDetIdRefinedEE*interestingEcalDetIdRefinedES*
                                          interestingTrackEcalDetIds*
                                          reducedEcalRecHitsEB*
                                          reducedEcalRecHitsEE*
                                          seldigis*
                                          reducedEcalRecHitsES)
                                          
reducedEcalRecHitsSequenceEcalOnly = cms.Sequence(interestingEcalDetIdEB*interestingEcalDetIdEBU*
                                          interestingEcalDetIdEE*
                                          reducedEcalRecHitsEB*
                                          reducedEcalRecHitsEE*
                                          seldigis)                                          
