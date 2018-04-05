import FWCore.ParameterSet.Config as cms

minbiasana = cms.EDAnalyzer("Analyzer_minbias",
                            HistOutFile = cms.untracked.string('analysis_minbias.root'),
                            hbheInputMB = cms.InputTag("hbherecoMB"),
                            hoInputMB = cms.InputTag("horecoMB"),
                            hfInputMB = cms.InputTag("hfrecoMB"),
                            hbheInputNoise = cms.InputTag("hbherecoNoise"),
                            hoInputNoise = cms.InputTag("horecoNoise"),
                            hfInputNoise = cms.InputTag("hfrecoNoise"),
                            Recalib = cms.bool(False)
                            )
