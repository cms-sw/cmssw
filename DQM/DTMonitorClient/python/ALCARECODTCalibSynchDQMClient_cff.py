import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtResolutionAnalysisTest_cfi import *
dtResolutionAnalysisTestAlca = dtResolutionAnalysisTest.clone() 
dtResolutionAnalysisTestAlca.topHistoFolder = 'AlCaReco/DtCalibSynch/01-Calibration'

ALCARECODTCalibSynchDQMClient = cms.Sequence(dtResolutionAnalysisTestAlca)


