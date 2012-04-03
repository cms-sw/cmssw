import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import siPixelDigis
siPixelDigisForLumi = siPixelDigis.clone()
siPixelDigisForLumi.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

from Configuration.StandardSequences.Reconstruction_cff import siPixelClusters
siPixelClustersForLumi = siPixelClusters.clone()
siPixelClustersForLumi.src = cms.InputTag("siPixelDigisForLumi")

# Sequence #
seqALCARECOLumiPixels = cms.Sequence(siPixelDigisForLumi + siPixelClustersForLumi)
