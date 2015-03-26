import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigisForLumi = siPixelDigis.clone()
siPixelDigisForLumi.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
siPixelClustersForLumi = siPixelClustersPreSplitting.clone()
siPixelClustersForLumi.src = cms.InputTag("siPixelDigisForLumi")

# Sequence #
seqALCARECOLumiPixels = cms.Sequence(siPixelDigisForLumi + siPixelClustersForLumi)
