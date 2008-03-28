import FWCore.ParameterSet.Config as cms

#
# Analysis sequences
#
# Electroweak Analysis
from ElectroWeakAnalysis.Configuration.ElectroWeakAnalysis_cff import *
from HiggsAnalysis.Configuration.HiggsAnalysis_cff import *
from TopQuarkAnalysis.Configuration.TopQuarkAnalysis_cff import *
analysis = cms.Sequence(electroWeakAnalysis)

