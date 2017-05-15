import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


SusyPostProcessor = DQMEDHarvester("SusyPostProcessor",
                                   folderName = cms.string("JetMET/SUSYDQM/"),
                                   quantile = cms.double(0.05)
                                   )

SusyPostProcessorSequence = cms.Sequence(SusyPostProcessor)
