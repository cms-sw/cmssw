import FWCore.ParameterSet.Config as cms

StripCPEfromTemplateESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTemplate'),
                                                  ComponentType = cms.string('StripCPEfromTemplate'),
                                                  UseTemplateReco            = cms.bool(False),
                                                  TemplateRecoSpeed          = cms.int32(0),
                                                  UseStripSplitClusterErrors = cms.bool(False)
                                                  )


