import FWCore.ParameterSet.Config as cms

StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTrackAngle'),
                                                  UseTemplateReco            = cms.bool(False),
                                                  TemplateRecoSpeed          = cms.int32(0),
                                                  UseStripSplitClusterErrors = cms.bool(False)
                                                  )


