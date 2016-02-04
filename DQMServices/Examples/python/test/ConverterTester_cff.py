import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMEnvironment_cfi import *
from DQMServices.Examples.test.ConverterTester_cfi import *
DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/ConverterTester/Workflow/RECO'

