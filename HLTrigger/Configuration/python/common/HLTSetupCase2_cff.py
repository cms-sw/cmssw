import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTanalyzers.hltGetRaw_cfi import *
hlt2GetRaw = copy.deepcopy(hltGetRaw)
from HLTrigger.Configuration.common.HLTSetupFromRaw_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from HLTrigger.Configuration.common.HLTSetupCommon_cff import *
hltBegin = cms.Sequence(hlt2GetRaw+L1HltSeed+offlineBeamSpot)

