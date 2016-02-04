import FWCore.ParameterSet.Config as cms


SusyPostProcessor = cms.EDAnalyzer("SusyPostProcessor",
                                   folderName = cms.string("JetMET/SUSYDQM/")
                                   )

SusyPostProcessorSequence = cms.Sequence(SusyPostProcessor)
