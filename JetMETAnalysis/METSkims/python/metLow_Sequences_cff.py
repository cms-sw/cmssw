import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.metLow_HLTPaths_cfi import *
metLowSkimHLTFilter = cms.Sequence(metPre1HLTFilter+metPre2HLTFilter+metPre3HLTFilter)

