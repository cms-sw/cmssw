import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import siPixelDigis
#siPixelDigisForLumi = siPixelDigis.clone()
siPixelDigis.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

from Configuration.StandardSequences.Reconstruction_cff import siPixelClusters
#siPixelClustersForLumi = siPixelClusters.clone()
#siPixelClustersForLumi.src = cms.InputTag("siPixelDigis")

# Sequence #
seqALCARECOLumiPixels = cms.Sequence(siPixelDigis + siPixelClusters)
