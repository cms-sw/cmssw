import FWCore.ParameterSet.Config as cms

DQMExample_Step2 = cms.EDAnalyzer("DQMExample_Step2",
                                  numMonitorName = cms.string("Physics/TopTest/ElePt_leading_HLT_matched"),
                                  denMonitorName = cms.string("Physics/TopTest/ElePt_leading")
                                  )