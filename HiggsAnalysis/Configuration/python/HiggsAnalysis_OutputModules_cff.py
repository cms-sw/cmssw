import FWCore.ParameterSet.Config as cms

#
# HiggsAnalysis output modules
#
# Dominique Fortin - UC Riverside
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_OutputModule_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_OutputModule_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_OutputModule_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_OutputModule_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_OutputModule_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_OutputModule_cff import *
# include "HiggsAnalysis/Skimming/data/rsTo2Gamma_OutputModule.cff"
HiggsAnalysisOutput = cms.Sequence(heavyChHiggsToTauNuOutputModuleRECOSIM+higgsTo2GammaOutputModuleRECOSIM+higgsToInvisibleOutputModuleRECOSIM+higgsToTauTauLeptonTauOutputModuleAODSIM+higgsToWW2LeptonsOutputModuleAODSIM+higgsToZZ4LeptonsOutputModuleRECOSIM)

