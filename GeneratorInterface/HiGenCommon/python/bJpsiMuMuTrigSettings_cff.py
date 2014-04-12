import FWCore.ParameterSet.Config as cms

bJpsiMuMuTrigCommon = cms.PSet(filterType = cms.untracked.string("PartonHadronDecayGenEvtSelector"),
                               partons      = cms.vint32(5),
                               partonStatus = cms.vint32(2),
                               partonEtaMax = cms.vdouble(999.),
                               partonPtMin  = cms.vdouble(0.),
                               hadrons      = cms.vint32(443),
                               hadronStatus = cms.vint32(2),
                               hadronEtaMax = cms.vdouble(999.),
                               hadronEtaMin = cms.vdouble(-999.),
                               hadronPMin   = cms.vdouble(0.),
                               hadronPtMax  = cms.vdouble(999.),
                               hadronPtMin  = cms.vdouble(0.),
                               decays      = cms.int32(13),
                               decayStatus = cms.int32(1),
                               decayEtaMax = cms.double(2.5),
                               decayEtaMin = cms.double(-2.5),
                               decayPMin   = cms.double(2.5),
                               decayPtMax  = cms.double(999),
                               decayPtMin  = cms.double(0.),
                               decayNtrig  = cms.int32(2)
                               )

bJpsiMuMuTrigPt03   = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(3.),
                                                hadronPtMin  = cms.vdouble(0.)
                                                )


bJpsiMuMuTrigPt36   = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(6.),
                                                hadronPtMin  = cms.vdouble(3.)
                                                )

bJpsiMuMuTrigPt69   = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(9.),
                                                hadronPtMin  = cms.vdouble(6.)
                                                )

bJpsiMuMuTrigPt912  = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(12.),
                                                hadronPtMin  = cms.vdouble(9.)
                                                )

bJpsiMuMuTrigPt1215 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(15.),
                                                hadronPtMin  = cms.vdouble(12.),
                                              )
bJpsiMuMuTrigPt1530 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                                hadronPtMin  = cms.vdouble(15.),
                                              )


bJpsiMuMuTrigPt1518 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(18.),
                                                hadronPtMin  = cms.vdouble(15.)
                                              )

bJpsiMuMuTrigPt1821 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(21.),
                                                hadronPtMin  = cms.vdouble(18.)
                                              )

bJpsiMuMuTrigPt2124 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(24.),
                                                hadronPtMin  = cms.vdouble(21.)
                                              )

bJpsiMuMuTrigPt2427 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(27.),
                                                hadronPtMin  = cms.vdouble(24.)
                                              )

bJpsiMuMuTrigPt2730 = bJpsiMuMuTrigCommon.clone(hadronPtMax  = cms.vdouble(30.),
                                                hadronPtMin  = cms.vdouble(27.)
                                              )











