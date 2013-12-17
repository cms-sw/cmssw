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

#SuperClusters for GED reco
import RecoEcal.EgammaClusterProducers.interestingDetIdFromSuperClusterProducer_cfi

interestingEcalDetIdPFEB = RecoEcal.EgammaClusterProducers.interestingDetIdFromSuperClusterProducer_cfi.interestingDetIdFromSuperClusterProducer.clone(
    superClustersLabel = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )
    
interestingEcalDetIdPFEE = RecoEcal.EgammaClusterProducers.interestingDetIdFromSuperClusterProducer_cfi.interestingDetIdFromSuperClusterProducer.clone(
    superClustersLabel = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )
    
interestingEcalDetIdRefinedEB = RecoEcal.EgammaClusterProducers.interestingDetIdFromSuperClusterProducer_cfi.interestingDetIdFromSuperClusterProducer.clone(
    superClustersLabel = cms.InputTag("particleFlowEGamma",""),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )
    
interestingEcalDetIdRefinedEE = RecoEcal.EgammaClusterProducers.interestingDetIdFromSuperClusterProducer_cfi.interestingDetIdFromSuperClusterProducer.clone(
    superClustersLabel = cms.InputTag("particleFlowEGamma",""),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
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
                                      EndcapSuperClusterCollection = cms.InputTag("particleFlowSuperClusterECAL",
                                      "particleFlowSuperClusterECALEndcapWithPreshower"),
                                      OutputLabel_ES = cms.string(''),
                                      interestingDetIds = cms.VInputTag()
                                      )

#selected digis
from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cff import *

reducedEcalRecHitsSequence = cms.Sequence(interestingEcalDetIdEB*interestingEcalDetIdEBU*
                                          interestingEcalDetIdEE*
                                          interestingEcalDetIdPFEB*interestingEcalDetIdPFEE*
                                          interestingEcalDetIdRefinedEB*interestingEcalDetIdRefinedEE*
                                          interestingTrackEcalDetIds*
                                          reducedEcalRecHitsEB*
                                          reducedEcalRecHitsEE*
                                          seldigis*
                                          reducedEcalRecHitsES)
