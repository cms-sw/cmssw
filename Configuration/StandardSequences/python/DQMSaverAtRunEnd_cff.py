import FWCore.ParameterSet.Config as cms

# needed backend
from DQMServices.Core.DQMStore_cfg import *

# needed output
from DQMServices.Components.DQMEnvironment_cfi import *


dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'


DQMSaver = cms.Sequence(dqmSaver)
