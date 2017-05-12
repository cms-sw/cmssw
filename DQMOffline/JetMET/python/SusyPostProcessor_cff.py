import FWCore.ParameterSet.Config as cms


SusyPostProcessor = cms.EDProducer("SusyPostProcessor",
                                   folderName = cms.string("JetMET/SUSYDQM/"),
                                   quantile = cms.double(0.05)
                                   )

SusyPostProcessorSequence = cms.Sequence(SusyPostProcessor)
