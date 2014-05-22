import FWCore.ParameterSet.Config as cms

StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTrackAngle'),
                                                  ComponentType = cms.string('StripCPEfromTrackAngle'),
                                                  LC_P0         = cms.double(-.326),
                                                  LC_P1         = cms.double( .618),
                                                  LC_P2         = cms.double( .300),
                                                  TIB_P0        = cms.double(-.742),
                                                  TIB_P1        = cms.double( .202),
                                                  TOB_P0        = cms.double(-1.026),
                                                  TOB_P1        = cms.double( .253),
                                                  TID_P0        = cms.double(-1.427),
                                                  TID_P1        = cms.double( .433),
                                                  TEC_P0        = cms.double(-1.885),
                                                  TEC_P1        = cms.double( .471),
                                                  )


