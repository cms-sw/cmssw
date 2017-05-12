import FWCore.ParameterSet.Config as cms

EgammaPostVal = cms.EDProducer("EmDQMPostProcessor",
   subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
   dataSet = cms.untracked.string("unknown"),                  
   noPhiPlots = cms.untracked.bool(True),                  
                              )
