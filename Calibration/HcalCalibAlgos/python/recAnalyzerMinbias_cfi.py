import FWCore.ParameterSet.Config as cms

RecAnalyzerMinbias = cms.EDAnalyzer("RecAnalyzerMinbias",
                                    hbheInputMB = cms.InputTag("hbherecoMB"),
                                    hfInputMB   = cms.InputTag("hfrecoMB"),
                                    RunNZS      = cms.bool(True),
                                    ELowHB      = cms.double(4),
                                    EHighHB     = cms.double(100),
                                    ELowHE      = cms.double(4),
                                    EHighHE     = cms.double(150),
                                    ELowHF      = cms.double(10),
                                    EHighHF     = cms.double(150),
                                    HistOutFile = cms.untracked.string('analysis_minbias.root'),
                                    CorrFile    = cms.untracked.string('CorFactor.txt'),
                                    IgnoreL1    = cms.untracked.bool(False),
                                    HcalIeta    = cms.untracked.vint32([]),
                                    HcalIphi    = cms.untracked.vint32([]),
                                    HcalDepth   = cms.untracked.vint32([]),
                                    )
