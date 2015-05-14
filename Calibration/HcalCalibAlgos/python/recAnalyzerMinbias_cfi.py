import FWCore.ParameterSet.Config as cms

RecAnalyzerMinbias = cms.EDAnalyzer("RecAnalyzerMinbias",
                                    HistOutFile = cms.untracked.string('analysis_minbias.root'),
                                    hbheInputMB = cms.InputTag("hbherecoMB"),
                                    hfInputMB   = cms.InputTag("hfrecoMB"),
                                    Recalib     = cms.bool(False),
                                    IgnoreL1    = cms.untracked.bool(False)
                                    )
