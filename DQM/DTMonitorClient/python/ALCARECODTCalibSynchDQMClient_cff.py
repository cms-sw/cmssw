import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtResolutionAnalysisTest_cfi import *
dtResolutionAnalysisTest.topHistoFolder = 'DtCalibSynch/01-Calibration'

ALCARECODTCalibSynchDQMClient = cms.Sequence(dtResolutionAnalysisTest)


