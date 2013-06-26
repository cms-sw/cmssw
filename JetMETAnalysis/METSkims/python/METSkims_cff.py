import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.metHigh_Sequences_cff import *
from JetMETAnalysis.METSkims.metLow_Sequences_cff import *
#include "JetMETAnalysis/METSkims/data/sumET_Sequences.cff"
metSkims = cms.Sequence(metHighSkimHLTFilter+metLowSkimHLTFilter)

