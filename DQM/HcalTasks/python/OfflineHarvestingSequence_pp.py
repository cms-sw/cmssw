import FWCore.ParameterSet.Config as cms
from DQM.HcalTasks.HcalOfflineHarvesting import *

# apply some customization
# -	ptype = 1 Offlien processing
# - runkey value 2 - cosmics
hcalOfflineHarvesting.ptype = 1
hcalOfflineHarvesting.runkeyVal = 0
hcalOfflineHarvesting.runkeyName = "pp_run"
