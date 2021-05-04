import FWCore.ParameterSet.Config as cms

# needed backend                                                                                                                                                                                                                             
from DQMServices.Core.DQMStore_cfg import *

# needed output                                                                                                                                                                                                                              
from DQMServices.Components.DQMEnvironment_cfi import *


dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'


dqmSaver.saveByRun = -1
dqmSaver.saveAtJobEnd = True  
dqmSaver.forceRunNumber = 999999

DQMSaver = cms.Sequence(dqmSaver)

# configuration is modified as a side effect, this is just a placeholder
# to allow using this file as a customisation for cmsDriver.
def customise(process):
    return process
