import FWCore.ParameterSet.Config as cms

shallowRechitClusters = cms.EDProducer("ShallowRechitClustersProducer",
                                       Prefix=cms.string("cluster"),
                                       Suffix=cms.string(""),
                                       Clusters=cms.InputTag("siStripClusters"),
                                       InputTags= cms.VInputTag(
    cms.InputTag('siStripMatchedRecHits:rphiRecHit'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHit'),
    cms.InputTag('siStripMatchedRecHits:rphiRecHitUnmatched'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHitUnmatched')
    )
                                       )
