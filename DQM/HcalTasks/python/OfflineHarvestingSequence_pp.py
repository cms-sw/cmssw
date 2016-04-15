import FWCore.ParameterSet.Config as cms
from DQM.HcalTasks.HcalOfflineHarvesting import *

# apply some customization
# -	ptype = 1 Offlien processing
# - runkey value 2 - cosmics
hcalOfflineHarvesting.ptype = cms.untracked.int32(1)
hcalOfflineHarvesting.runkeyVal = cms.untracked.int32(0)
hcalOfflineHarvesting.runkeyName = cms.untracked.string("pp_run")
