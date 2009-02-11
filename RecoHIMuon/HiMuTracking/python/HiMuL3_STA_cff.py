import FWCore.ParameterSet.Config as cms
muonFilter = cms.EDFilter("TestMuL1L2Filter",
    #CandTag = cms.InputTag("standAloneMuons"),
    L2CandTag = cms.InputTag("standAloneMuons"),
    PrimaryVertexTag = cms.InputTag("pixelVertices"),
    NavigationPSet = cms.PSet(
        ComponentName = cms.string('SimpleNavigationSchool')
    ),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    TTRHBuilder = cms.string('WithoutRefit')
)
