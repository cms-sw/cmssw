import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQM.SiTrackerPhase2.Phase2OTMonitorRecHit_cfi import Phase2OTMonitorRecHit

rechitMonitorOT = Phase2OTMonitorRecHit.clone()                                         
