import FWCore.ParameterSet.Config as cms

jpsiMuMuTrigCommon = cms.PSet(filterType = cms.untracked.string("HadronDecayGenEvtSelector"),
                              hadrons      = cms.vint32(443),
                              hadronStatus = cms.vint32(2),
                              decays      = cms.int32(13),
                              decayStatus = cms.int32(1),
                              hadronEtaMax = cms.vdouble(999.),
                              hadronEtaMin = cms.vdouble(-999.,),
                              hadronPMin   = cms.vdouble(0.),
                              hadronPtMax  = cms.vdouble(999.),
                              hadronPtMin  = cms.vdouble(0.),
                              decayEtaMax = cms.double(2.5),
                              decayEtaMin = cms.double(-2.5),
                              decayPMin   = cms.double(2.5),
                              decayPtMax  = cms.double(999),
                              decayPtMin  = cms.double(0.),
                              decayNtrig  = cms.int32(2)
                              )

jpsiMuMuTrigPt03   = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(3.),
                                              hadronPtMin  = cms.vdouble(0.)
                                              )


jpsiMuMuTrigPt36   = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(6.),
                                              hadronPtMin  = cms.vdouble(3.)
                                              )

jpsiMuMuTrigPt69   = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(9.),
                                              hadronPtMin  = cms.vdouble(6.)
                                              )

jpsiMuMuTrigPt912  = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(12.),
                                              hadronPtMin  = cms.vdouble(9.)
                                              )

jpsiMuMuTrigPt1215 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(15.),
                                              hadronPtMin  = cms.vdouble(12.),
                                              )
jpsiMuMuTrigPt1530 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                              hadronPtMin  = cms.vdouble(15.),
                                              )


jpsiMuMuTrigPt1518 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(18.),
                                              hadronPtMin  = cms.vdouble(15.)
                                              )

jpsiMuMuTrigPt1821 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(21.),
                                              hadronPtMin  = cms.vdouble(18.)
                                              )

jpsiMuMuTrigPt2124 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(24.),
                                              hadronPtMin  = cms.vdouble(21.)
                                              )

jpsiMuMuTrigPt2427 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(27.),
                                              hadronPtMin  = cms.vdouble(24.)
                                              )

jpsiMuMuTrigPt2730 = jpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                              hadronPtMin  = cms.vdouble(27.)
                                              )








