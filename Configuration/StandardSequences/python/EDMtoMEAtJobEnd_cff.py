import FWCore.ParameterSet.Config as cms

from DQMServices.Components.EDMtoMEConverter_cff import *

DQMStore.referenceFileName = ''
dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'

DQMStore.collateHistograms = True
EDMtoMEConverter.convertOnEndLumi = True
EDMtoMEConverter.convertOnEndRun = True
dqmSaver.saveByRun = -1
dqmSaver.saveAtJobEnd = True  
dqmSaver.forceRunNumber = 1

EDMtoME = cms.Sequence(EDMtoMEConverter)

DQMSaver = cms.Sequence(dqmSaver)
