import FWCore.ParameterSet.Config as cms

upsilon1sMuMuTrigCommon = cms.PSet(filterType = cms.untracked.string("HadronDecayGenEvtSelector"),
                                   hadrons      = cms.vint32(553),
                                   hadronStatus = cms.vint32(2),
                                   decays       = cms.int32(13),
                                   decayStatus  = cms.int32(1),
                                   hadronEtaMax = cms.vdouble(999.),
                                   hadronEtaMin = cms.vdouble(-999.,),
                                   hadronPMin   = cms.vdouble(0.),
                                   hadronPtMax  = cms.vdouble(999.),
                                   hadronPtMin  = cms.vdouble(0.),
                                   decayEtaMax  = cms.double(2.5),
                                   decayEtaMin  = cms.double(-2.5),
                                   decayPMin    = cms.double(2.5),
                                   decayPtMax   = cms.double(999.),
                                   decayPtMin   = cms.double(0.),
                                   decayNtrig  = cms.int32(2)
                                   )

upsilon1sMuMuTrigPt03   = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(3.),
                                                        hadronPtMin  = cms.vdouble(0.)
                                                        )


upsilon1sMuMuTrigPt36   = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(6.),
                                                        hadronPtMin  = cms.vdouble(3.)
                                                        )

upsilon1sMuMuTrigPt69   = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(9.),
                                                        hadronPtMin  = cms.vdouble(6.)
                                                        )

upsilon1sMuMuTrigPt912  = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(12.),
                                                        hadronPtMin  = cms.vdouble(9.)
                                                        )

upsilon1sMuMuTrigPt1215 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(15.),
                                                        hadronPtMin  = cms.vdouble(12.),
                                                        )

upsilon1sMuMuTrigPt1530 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                                        hadronPtMin  = cms.vdouble(15.),
                                                        )


upsilon1sMuMuTrigPt1518 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(18.),
                                                        hadronPtMin  = cms.vdouble(15.)
                                                        )

upsilon1sMuMuTrigPt1821 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(21.),
                                                        hadronPtMin  = cms.vdouble(18.)
                                                        )

upsilon1sMuMuTrigPt2124 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(24.),
                                                        hadronPtMin  = cms.vdouble(21.)
                                                        )

upsilon1sMuMuTrigPt2427 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(27.),
                                                        hadronPtMin  = cms.vdouble(24.)
                                                        )

upsilon1sMuMuTrigPt2730 = upsilon1sMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                                        hadronPtMin  = cms.vdouble(27.)
                                                        )
