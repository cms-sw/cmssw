import FWCore.ParameterSet.Config as cms

hltTauQualityTesterForZ8E29 = cms.EDFilter("HLTTauRelvalQTester",
                                        qtList = cms.untracked.FileInPath('HLTriggerOffline/Tau/data/QTDefault.xml'),
                                        reportThreshold         = cms.untracked.string('black'),
                                        prescaleFactor          = cms.untracked.int32(1),
                                        getQualityTestsFromFile = cms.untracked.bool(True),
                                        qtestOnEndJob           = cms.untracked.bool(True),
                                        qtestOnEndLumi          = cms.untracked.bool(False),
                                        testInEventloop         = cms.untracked.bool(False),
                                        verboseQT               = cms.untracked.bool(True),
                                        refMothers              = cms.InputTag("TauMCProducer","Mothers"),
                                        mothers                 = cms.vint32(23)
                                    )


#hltTauQualityTesterForZ8E29 = cms.EDFilter("QualityTester",
#                                        qtList = cms.untracked.FileInPath('HLTriggerOffline/Tau/data/QTDefault.xml'),
#                                        reportThreshold         = cms.untracked.string('black'),
#                                        prescaleFactor          = cms.untracked.int32(1),
#                                        getQualityTestsFromFile = cms.untracked.bool(True),
#                                        qtestOnEndJob           = cms.untracked.bool(True),
#                                        qtestOnEndLumi          = cms.untracked.bool(False),
#                                        testInEventloop         = cms.untracked.bool(False),
#                                        verboseQT               = cms.untracked.bool(True)#,
#                                        refMothers              = cms.InputTag("TauMCProducer","Mothers"),
#                                        mothers                 = cms.vint32(23)
#                                    )
