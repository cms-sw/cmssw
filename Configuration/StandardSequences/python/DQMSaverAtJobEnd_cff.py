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
dqmSaver.forceRunNumber = 999999

DQMSaver = cms.Sequence(dqmSaver)

# configuration is modified as a side effect, this is just a placeholder
# to allow using this file as a customisation for cmsDriver.
def customise(process):
    return process
