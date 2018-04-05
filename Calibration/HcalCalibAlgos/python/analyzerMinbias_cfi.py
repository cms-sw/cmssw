import FWCore.ParameterSet.Config as cms

AnalyzerMinbias  = cms.EDAnalyzer("AnalyzerMinbias",                         
                                  hbheInputMB    = cms.InputTag("hbherecoMB"),
                                  hoInputMB      = cms.InputTag("horecoMB"),
                                  hfInputMB      = cms.InputTag("hfrecoMB"),
                                  hbheInputNoise = cms.InputTag("hbherecoNoise"),
                                  hoInputNoise   = cms.InputTag("horecoNoise"),
                                  hfInputNoise   = cms.InputTag("hfrecoNoise"),
                                  Recalib        = cms.bool(False),
                                  IgnoreL1       = cms.untracked.bool(True),
                                  RunNZS         = cms.untracked.bool(False),
                                  HistOutFile    = cms.untracked.string('analysis_minbias.root'),           
                                  )                                   
