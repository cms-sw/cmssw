import FWCore.ParameterSet.Config as cms

# needed backend                                                                                                                                                                                                                             
from DQMServices.Core.DQMStore_cfg import *

# needed output                                                                                                                                                                                                                              
from DQMServices.Components.DQMEnvironment_cfi import *


DQMStore.referenceFileName = ''
dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'

DQMStore.collateHistograms = True

dqmSaver.saveByRun = -1
dqmSaver.saveAtJobEnd = True  
dqmSaver.forceRunNumber = 1

DQMSaver = cms.Sequence(dqmSaver)
