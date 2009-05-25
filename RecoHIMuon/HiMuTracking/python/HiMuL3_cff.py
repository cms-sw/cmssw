import FWCore.ParameterSet.Config as cms
muonFilter = cms.EDFilter("TestMuL1L2Filter",
    CandTag = cms.InputTag("standAloneMuons"),
    NavigationPSet = cms.PSet(
        ComponentName = cms.string('SimpleNavigationSchool')
    ),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
)
