import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.l1ExtraDQM_cfi import *

# import also the L1Extra producer, configured to run for all BX
from L1Trigger.Configuration.L1Extra_cff import *
l1extraParticles.centralBxOnly = cms.bool(False)
