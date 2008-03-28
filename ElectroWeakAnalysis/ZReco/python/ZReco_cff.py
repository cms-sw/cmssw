import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuGoldenSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToEESequences_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_DoubleTauSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_EMuSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_MuTauSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_ETauSequences_cff import *
zReco = cms.Sequence(cms.SequencePlaceholder("zToMuMuAnalysisSequence")+zToMuMuGoldenAnalysisSequence+zToEEAnalysisSequence+zToTauTau_DoubleTauAnalysis+zToTauTau_EMuAnalysis+zToTauTau_MuTauAnalysis+zToTauTau_ETauAnalysis)

