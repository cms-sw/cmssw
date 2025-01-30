# ------------------------------------------- #
# Scouting DQM sequence for offline DQM       #
#                                             #
# used by DQM GUI: DQMOffline/Configuration   #
# ------------------------------------------- #
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester 

from HLTriggerOffline.Scouting.HLTScoutingEGammaDqmOffline_cff import *

hltScoutingDqmOffline = cms.Sequence(hltScoutingEGammaDqmOffline)

