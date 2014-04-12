import FWCore.ParameterSet.Config as cms

StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTrackAngle'),
                                                  ComponentType = cms.string('StripCPEfromTrackAngle')
                                                  )


