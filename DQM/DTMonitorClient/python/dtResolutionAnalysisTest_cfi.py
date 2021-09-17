import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtResolutionAnalysisTest = DQMEDHarvester("DTResolutionAnalysisTest",
                                          diagnosticPrescale = cms.untracked.int32(1),
                                          maxGoodMeanValue = cms.untracked.double(0.005),
                                          minBadMeanValue = cms.untracked.double(0.015),
                                          maxGoodSigmaValue = cms.untracked.double(0.05),
                                          minBadSigmaValue = cms.untracked.double(0.08),
                                          # top folder for the histograms in DQMStore
                                          topHistoFolder = cms.untracked.string('DT/02-Segments')
                                          )


