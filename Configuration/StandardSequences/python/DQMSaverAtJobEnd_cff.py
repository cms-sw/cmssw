import FWCore.ParameterSet.Config as cms

# needed backend                                                                                                                                                                                                                             
from DQMServices.Core.DQMStore_cfg import *

# needed output                                                                                                                                                                                                                              
from DQMServices.Components.DQMEnvironment_cfi import *

# modifications for run-dependent MC
from Configuration.ProcessModifiers.runDependent_cff import runDependent

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'


dqmSaver.saveByRun = -1
dqmSaver.saveAtJobEnd = True  
dqmSaver.forceRunNumber = 999999

DQMSaver = cms.Sequence(dqmSaver)

runDependent.toModify(dqmSaver, forceRunNumber = 1)

# configuration is modified as a side effect, this is just a placeholder
# to allow using this file as a customisation for cmsDriver.
def customise(process):
    return process
