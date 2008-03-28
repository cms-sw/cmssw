import FWCore.ParameterSet.Config as cms

#
# HeavyFlavorAnalysis standard sequences
#
from HeavyFlavorAnalysis.Skimming.onia_Sequences_cff import *
from HeavyFlavorAnalysis.Skimming.tauTo3MuSequences_cff import *
heavyFlavorAnalysis = cms.Sequence(cms.SequencePlaceholder("onia")+cms.SequencePlaceholder("tauTo3Mu"))

