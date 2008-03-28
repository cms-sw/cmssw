import FWCore.ParameterSet.Config as cms

#
# HiggsAnalysis standard sequences
#
# Dominique Fortin - UC Riverside
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_Sequences_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_Sequences_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_Sequences_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_Sequences_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_Sequences_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_Sequences_cff import *
# include "HiggsAnalysis/Skimming/data/rsTo2Gamma_Sequences.cff"
higgsAnalysis = cms.Sequence(heavyChHiggsToTauNuSequence+higgsTo2GammaSequence+higgsToInvisibleSequence+higgsToTauTauLeptonTauSequence+higgsToWW2LeptonsSequence+higgsToZZ4LeptonsSequence)

