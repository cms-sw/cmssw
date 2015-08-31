import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise

GlobalTag = gtCustomise(GlobalTag, 'auto:run2_data', '')
