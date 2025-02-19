import FWCore.ParameterSet.Config as cms

siStripClustersDSV = cms.EDProducer(
    "SiStripClustersDSVBuilder",
    SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility"),
    SiStripRefGetter  = cms.InputTag("siStripClusters"),
    DetSetVectorNew   = cms.untracked.bool(True)
    )

