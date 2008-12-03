import FWCore.ParameterSet.Config as cms

siStripClustersDSV = cms.EDFilter("SiStripClustersDSVBuilder",
    SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility"),
    SiStripRefGetter = cms.InputTag("siStripClusters")
)


