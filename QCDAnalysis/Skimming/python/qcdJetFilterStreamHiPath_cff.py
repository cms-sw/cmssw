import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.qcdJetFilterStreamHiSkim_cff import *
qcdJetFilterStreamHiPath = cms.Path(cms.SequencePlaceholder("qcdSingleJetFilterStreamHi"))

