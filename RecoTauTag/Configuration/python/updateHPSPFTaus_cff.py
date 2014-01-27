import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus that need to be rerun in order to update Monte Carlo/Data samples produced with CMSSW_7_0_x RecoTauTag tags
to the latest tau id. developments recommended by the Tau POG

authors: Evan Friis, Wisconsin
         Christian Veelken, LLR

'''

updateHPSPFTaus = cms.Sequence()
