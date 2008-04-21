import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
from HLTrigger.Configuration.common.CaloTowers_cff import *
from HLTrigger.Configuration.common.TrackerTracks_cff import *
from HLTrigger.Configuration.common.Vertexing_cff import *
from HLTrigger.Configuration.common.HLTFullReco_cff import *
import copy
from HLTrigger.HLTfilters.hltBool_cfi import *
boolEnd = copy.deepcopy(hltBool)
from HLTrigger.Configuration.common.HLTEndpath_cff import *
hltEnd = cms.Sequence(boolEnd)
boolEnd.result = True

