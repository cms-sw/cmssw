import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.sumET_HLTPaths_cfi import *
sumETSkimHLTFilter = cms.Sequence(sumETHLTFilter)

