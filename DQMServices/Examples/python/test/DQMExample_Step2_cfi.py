import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

DQMExampleStep2 = DQMEDHarvester("DQMExample_Step2",
                                  numMonitorName = cms.string("Physics/TopTest/ElePt_leading_HLT_matched"),
                                  denMonitorName = cms.string("Physics/TopTest/ElePt_leading")
                                  )
