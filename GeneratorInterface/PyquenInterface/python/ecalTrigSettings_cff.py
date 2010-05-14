
import FWCore.ParameterSet.Config as cms

ecalTrigCommon = cms.PSet(filterType = cms.untracked.string("EcalGenEvtSelector"),
                          etaMax = cms.double(3),
                          partons = cms.vint32(1,2,3,4,5,6,21,22),
                          partonStatus = cms.vint32(2,2,2,2,2,2,2,1),
                          particleStatus = cms.vint32(2, 2, 2, 2, 2,
                                                                2, 1, 1, 2, 2,
                                                                1, 1, 1),
                          particles = cms.vint32(221, -221, # eta
                                                           331, -331, # eta'
                                                           223, -223, # omega
                                                           211, -211, # pi
                                                           111,       # pi0
                                                           311,       # K0
                                                           11, -11,   # e
                                                           22         # gamma
                                                           )
                          
                          )

ecalTrigPt20 = cms.PSet(ecalTrigCommon,
                        partonPt = cms.vdouble(38.5,38.5,38.5,38.5,38.5,38.5,38.5,38.5),
                        particlePt = cms.vdouble(17.5, 17.5, # eta
                                                           16., 16., # eta'
                                                           18., 18., # omega
                                                           14.5, 14.5, # pi
                                                           17.5,      # pi0
                                                           15,    # K0
                                                           17.5, 17.5,   # e
                                                           38.5       # gamma
                                                           )
                        )


ecalTrigPt40 = cms.PSet(ecalTrigCommon,
                        partonPt = cms.vdouble(38.5,38.5,38.5,38.5,38.5,38.5,38.5,38.5),
                        particlePt = cms.vdouble(17.5, 17.5, # eta
                                                           16., 16., # eta'
                                                           18., 18., # omega
                                                           14.5, 14.5, # pi
                                                           17.5,      # pi0
                                                           15,    # K0
                                                           17.5, 17.5,   # e
                                                           38.5       # gamma
                                                           )
                        )

ecalTrigPt70 = cms.PSet(ecalTrigCommon,
                        partonPt = cms.vdouble(38.5,38.5,38.5,38.5,38.5,38.5,38.5,38.5),
                        particlePt = cms.vdouble(17.5, 17.5, # eta
                                                           16., 16., # eta'
                                                           18., 18., # omega
                                                           14.5, 14.5, # pi
                                                           17.5,      # pi0
                                                           15,    # K0
                                                           17.5, 17.5,   # e
                                                           38.5       # gamma
                                                           )
                        )



ecalTrigPt100 = cms.PSet(ecalTrigCommon,
                         partonPt = cms.vdouble(38.5,38.5,38.5,38.5,38.5,38.5,38.5,38.5),
                         particlePt = cms.vdouble(36.125, 36.125, # eta
                                                            35.25, 35.25, # eta'
                                                            34.75, 34.75, # omega
                                                            32.25, 32.25, # pi
                                                            32.25,        # pi0
                                                            31.5,         # K0
                                                            38.75, 38.75, # e
                                                            71.625       # gamma
                                                            )                         
                         )










