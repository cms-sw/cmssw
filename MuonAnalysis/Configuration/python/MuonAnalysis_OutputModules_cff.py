import FWCore.ParameterSet.Config as cms

from MuonAnalysis.Configuration.muonL1OutputModule_cfi import *
MuonAnalysisOutput = cms.Sequence(muonL1OutputModuleAODSIM+muonL1OutputModuleRECOSIM)

