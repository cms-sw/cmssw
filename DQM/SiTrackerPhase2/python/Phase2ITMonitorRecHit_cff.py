import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQM.SiTrackerPhase2.Phase2ITMonitorRecHit_cfi import Phase2ITMonitorRecHit 

rechitMonitorIT = Phase2ITMonitorRecHit.clone()
