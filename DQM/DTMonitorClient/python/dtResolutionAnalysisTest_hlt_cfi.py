import FWCore.ParameterSet.Config as cms

dtResolutionTestHLT = cms.EDAnalyzer("DTResolutionAnalysisTest",
                                     diagnosticPrescale = cms.untracked.int32(1),
                                     permittedMeanRange = cms.untracked.double(0.01),
                                     permittedSigmaRange = cms.untracked.double(0.08),
                                     # top folder for the histograms in DQMStore
                                     topHistoFolder = cms.untracked.string("HLT/HLTMonMuon/DT-Segments")
                                     )


