import FWCore.ParameterSet.Config as cms

#DQM services
from DQMServices.Components.MEtoEDMConverter_cfi import *
#Subsystem Includes 
# Tracker Alignment
from DQMOffline.Configuration.ALCARECOTkAlDQM_cff import *
DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

# unfortunally the ALCARECOTkAl-Producers can not go here because they are filters.
pathALCARECODQM = cms.Path(MEtoEDMConverter)

