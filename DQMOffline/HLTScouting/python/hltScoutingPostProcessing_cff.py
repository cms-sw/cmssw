import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from HLTriggerOffline.Scouting.HLTScoutingEGammaPostProcessing_cff import *

hltScoutingPostProcessing = cms.Sequence(hltScoutingEGammaPostProcessing)

