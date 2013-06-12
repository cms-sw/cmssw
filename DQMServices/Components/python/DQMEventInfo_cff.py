import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMEventInfo_cfi import *
dqmEnvCommon = dqmEnv.clone()
dqmEnvCommon.subSystemFolder=cms.untracked.string('Info')
