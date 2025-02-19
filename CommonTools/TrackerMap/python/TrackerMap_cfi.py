import FWCore.ParameterSet.Config as cms

TrackerMapTest = cms.EDFilter("TrackerGeometryTest",
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(False),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('CommonTools/TrackerMap/data/')
    )
)


