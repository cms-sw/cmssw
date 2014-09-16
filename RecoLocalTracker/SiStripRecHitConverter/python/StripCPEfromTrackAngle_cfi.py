import FWCore.ParameterSet.Config as cms

StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTrackAngle'),
                                                  ComponentType = cms.string('StripCPEfromTrackAngle'),
                                                  mLC_P0         = cms.double(-.326),
                                                  mLC_P1         = cms.double( .618),
                                                  mLC_P2         = cms.double( .300),
                                                  mTIB_P0        = cms.double(-.742),
                                                  mTIB_P1        = cms.double( .202),
                                                  mTOB_P0        = cms.double(-1.026),
                                                  mTOB_P1        = cms.double( .253),
                                                  mTID_P0        = cms.double(-1.427),
                                                  mTID_P1        = cms.double( .433),
                                                  mTEC_P0        = cms.double(-1.885),
                                                  mTEC_P1        = cms.double( .471),
                                                  useLegacyError = cms.bool(False),
                                                  )


