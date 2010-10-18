import FWCore.ParameterSet.Config as cms

# HLT photon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPhotonHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltPhotonHI.HLTPaths = ["HLT_HIPhoton15"]
hltPhotonHI.throw = False
hltPhotonHI.andOr = True

# selection of good superclusters
superClusterMerger =  cms.EDProducer("EgammaSuperClusterMerger",
  src = cms.VInputTag(
    cms.InputTag('correctedHybridSuperClusters'),
    cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
  )
)

superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
  src = cms.InputTag("superClusterMerger"),
  particleType = cms.string('gamma')
)

goodSuperClusters = cms.EDFilter("CandViewRefSelector",
  src = cms.InputTag("superClusterCands"),
  cut = cms.string('et > 20.0'),
  filter = cms.bool(True)
)

# select supercluster pairs around Z mass
superClusterCombiner = cms.EDFilter("CandViewShallowCloneCombiner",
  checkCharge = cms.bool(False),
  cut = cms.string('60 < mass < 120'),
  decay = cms.string('goodSuperClusters goodSuperClusters')
)

superClusterPairCounter = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag("superClusterCombiner"),
  minNumber = cms.uint32(1)
)

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(
    hltPhotonHI *
    superClusterMerger *
    superClusterCands *
    goodSuperClusters *
    superClusterCombiner *
    superClusterPairCounter
    )
