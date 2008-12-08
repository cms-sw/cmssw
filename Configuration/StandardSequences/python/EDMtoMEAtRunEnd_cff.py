import FWCore.ParameterSet.Config as cms

from DQMServices.Components.EDMtoMEConverter_cff import *

options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
    )

DQMStore.referenceFileName = ''
dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Global/CMSSW_X_Y_Z/RECO'

DQMStore.collateHistograms = False
EDMtoMEConverter.convertOnEndLumi = True
EDMtoMEConverter.convertOnEndRun = False


EDMtoME = cms.Sequence(EDMtoMEConverter)

DQMSaver = cms.Sequence(dqmSaver)
