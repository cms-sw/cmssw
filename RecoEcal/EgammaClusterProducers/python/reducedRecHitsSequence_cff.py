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

interestingEcalDetIdOOTPFEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterOOTECAL","particleFlowBasicClusterOOTECALBarrel"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

interestingEcalDetIdOOTPFEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterOOTECAL","particleFlowBasicClusterOOTECALEndcap"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )

interestingEcalDetIdOOTPFES = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("particleFlowSuperClusterOOTECAL","particleFlowBasicClusterOOTECALPreshower"),
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
            # oot
            cms.InputTag("interestingEcalDetIdOOTPFEB"),
            # egamma
            cms.InputTag("interestingGedEleIsoDetIdEB"),
            cms.InputTag("interestingGedGamIsoDetIdEB"),
            cms.InputTag("interestingOotGamIsoDetIdEB"),
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
            # oot
            cms.InputTag("interestingEcalDetIdOOTPFEE"),
            # egamma
            cms.InputTag("interestingGedEleIsoDetIdEE"),
            cms.InputTag("interestingGedGamIsoDetIdEE"),
            cms.InputTag("interestingOotGamIsoDetIdEE"),
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
                                        cms.InputTag("interestingEcalDetIdOOTPFES"),
                                      ),
                                      interestingDetIdsNotToClean = cms.VInputTag(
                                        cms.InputTag("interestingGedEgammaIsoESDetId"),
                                        cms.InputTag("interestingOotEgammaIsoESDetId"),
                                      )
)

#selected digis
from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cff import *

reducedEcalRecHitsTask = cms.Task(interestingEcalDetIdEB,interestingEcalDetIdEBU,
                                          interestingEcalDetIdEE,
                                          interestingEcalDetIdPFEB,interestingEcalDetIdPFEE,interestingEcalDetIdPFES,
                                          interestingEcalDetIdOOTPFEB,interestingEcalDetIdOOTPFEE,interestingEcalDetIdOOTPFES,
                                          interestingEcalDetIdRefinedEB,interestingEcalDetIdRefinedEE,interestingEcalDetIdRefinedES,
                                          interestingTrackEcalDetIds,
                                          reducedEcalRecHitsEB,
                                          reducedEcalRecHitsEE,
                                          seldigisTask,
                                          reducedEcalRecHitsES)
reducedEcalRecHitsSequence = cms.Sequence(reducedEcalRecHitsTask)
                                          
reducedEcalRecHitsSequenceEcalOnlyTask = cms.Task(interestingEcalDetIdEB,interestingEcalDetIdEBU,
                                          interestingEcalDetIdEE,
                                          reducedEcalRecHitsEB,
                                          reducedEcalRecHitsEE,
                                          seldigisTask)
reducedEcalRecHitsSequenceEcalOnly = cms.Sequence(reducedEcalRecHitsSequenceEcalOnlyTask)

_phase2_reducedEcalRecHitsTask = reducedEcalRecHitsTask.copy()
_phase2_reducedEcalRecHitsTask.remove(reducedEcalRecHitsES)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith( reducedEcalRecHitsTask , _phase2_reducedEcalRecHitsTask )


_fastSim_reducedEcalRecHitsTask = reducedEcalRecHitsTask.copyAndExclude(seldigisTask)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith( reducedEcalRecHitsTask, _fastSim_reducedEcalRecHitsTask)

_pp_on_AA_reducedEcalRecHitsTask = reducedEcalRecHitsTask.copy()
_pp_on_AA_reducedEcalRecHitsTask.remove(interestingEcalDetIdOOTPFEB)
_pp_on_AA_reducedEcalRecHitsTask.remove(interestingEcalDetIdOOTPFEE)
_pp_on_AA_reducedEcalRecHitsTask.remove(interestingEcalDetIdOOTPFES)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(reducedEcalRecHitsTask, _pp_on_AA_reducedEcalRecHitsTask)

pp_on_AA_2018.toModify(reducedEcalRecHitsEB.interestingDetIdCollections, func = lambda list: list.remove(cms.InputTag("interestingEcalDetIdOOTPFEB")) )
pp_on_AA_2018.toModify(reducedEcalRecHitsEB.interestingDetIdCollections, func = lambda list: list.remove(cms.InputTag("interestingOotGamIsoDetIdEB")) )
pp_on_AA_2018.toModify(reducedEcalRecHitsEE.interestingDetIdCollections, func = lambda list: list.remove(cms.InputTag("interestingEcalDetIdOOTPFEE")) )
pp_on_AA_2018.toModify(reducedEcalRecHitsEE.interestingDetIdCollections, func = lambda list: list.remove(cms.InputTag("interestingOotGamIsoDetIdEE")) )
pp_on_AA_2018.toModify(reducedEcalRecHitsES.interestingDetIds, func = lambda list: list.remove(cms.InputTag("interestingEcalDetIdOOTPFES")) )
pp_on_AA_2018.toModify(reducedEcalRecHitsES.interestingDetIdsNotToClean, func = lambda list: list.remove(cms.InputTag("interestingOotEgammaIsoESDetId")) )
