import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.softJetsSkim_cff import *
softJetsPath = cms.Path(singleJetTrigger*~muonTrigger*~electronTrigger*~photonTrigger)

