import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtResolutionAnalysisTest_cfi import *
dtResolutionAnalysisTestAlcaCosmics = dtResolutionAnalysisTest.clone( 
   topHistoFolder = 'AlCaReco/DtCalibSynchCosmics/01-Calibration'
)
ALCARECODTCalibSynchCosmicsDQMClient = cms.Sequence(dtResolutionAnalysisTestAlcaCosmics)
# foo bar baz
# fkSkMQNy4Vrkb
