import FWCore.ParameterSet.Config as cms

l1tGTTFileReader = cms.EDProducer('GTTFileReader',
  files = cms.vstring("gttOutput_0.txt"), #, "gttOutput_1.txt"),
  format = cms.untracked.string("APx")
)
