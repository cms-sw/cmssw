import FWCore.ParameterSet.Config as cms

shallowSimhitClusters = cms.EDProducer("ShallowSimhitClustersProducer",
                                       Prefix = cms.string("sim"),
                                       Clusters = cms.InputTag("siStripClusters"),
                                       InputTags = cms.VInputTag(
    cms.InputTag('g4SimHits:TrackerHitsTECHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTECLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIDHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIDLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIBHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIBLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTOBHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTOBLowTof')
    ))
