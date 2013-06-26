import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.qcdJetFilterStreamLoSkim_cff import *
qcdJetFilterStreamLoPath = cms.Path(cms.SequencePlaceholder("qcdSingleJetFilterStreamLo"))

