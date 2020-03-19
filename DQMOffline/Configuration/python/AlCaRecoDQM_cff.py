import FWCore.ParameterSet.Config as cms

#DQM services
#be sure to include the standard sequences
from DQMServices.Components.MEtoEDMConverter_cfi import *
#Subsystem Includes 
# Tracker Alignment
from DQMOffline.Alignment.ALCARECOTkAlDQM_cff import *
#Tracker Calibration
from DQMOffline.Configuration.ALCARECOTkCalDQM_cff import *
# Ecal Calibration
from DQMOffline.Configuration.ALCARECOEcalCalDQM_cff import *
# Hcal Calibration
from DQMOffline.Configuration.ALCARECOHcalCalDQM_cff import *
# Muon Calibration
from DQMOffline.Configuration.ALCARECOMuonDQM_cff import *

# unfortunally the ALCARECOTkAl-Producers can not go here because they are filters.
pathALCARECODQM = cms.Path(MEtoEDMConverter)

