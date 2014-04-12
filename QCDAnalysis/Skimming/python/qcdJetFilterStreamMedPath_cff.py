import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.qcdJetFilterStreamMedSkim_cff import *
qcdJetFilterStreamMedPath = cms.Path(cms.SequencePlaceholder("qcdSingleJetFilterStreamMed"))

