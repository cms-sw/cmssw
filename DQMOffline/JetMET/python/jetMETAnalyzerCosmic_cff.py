import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzer_cfi import *

jetMETAnalyzerCosmicSequence = cms.Sequence(jetMETAnalyzer)
