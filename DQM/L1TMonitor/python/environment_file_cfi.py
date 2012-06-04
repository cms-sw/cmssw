import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *

DQM.collectorHost = 'lxplus419.cern.ch'
DQM.collectorPort = 9190

from DQMServices.Components.DQMEnvironment_cfi import *

dqmSaver.convention = 'Online'
dqmSaver.referenceHandling = 'all'
dqmSaver.dirName = '.'
dqmSaver.producer = 'DQM'
dqmSaver.saveByTime = -1
dqmSaver.saveByLumiSection = -1
dqmSaver.saveByMinute = 8
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = True

