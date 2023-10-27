import FWCore.ParameterSet.Config as cms

l1tGTTFileReader = cms.EDProducer('GTTFileReader',
  files = cms.vstring("gttOutput_0.txt"), #, "gttOutput_1.txt"),
  l1VertexCollectionName = cms.string("L1VerticesFirmware"),
  format = cms.untracked.string("APx")
)
