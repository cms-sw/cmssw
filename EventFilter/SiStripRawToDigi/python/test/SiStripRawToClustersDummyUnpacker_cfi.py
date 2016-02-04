import FWCore.ParameterSet.Config as cms

SiStripUnpacker = cms.EDFilter("SiStripRawToClustersDummyUnpacker",
    SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility"),
    SiStripRefGetter = cms.InputTag("siStripClusters")
)


