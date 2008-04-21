import FWCore.ParameterSet.Config as cms

# Layer 0 default sequence 
# full cleaning
from PhysicsTools.PatAlgos.cleaningLayer0.caloJetCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.electronCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.pfTauCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.photonCleaner_cfi import *
# high level reco tasks needed before Layer0 cleaners
from PhysicsTools.PatAlgos.recoLayer0.beforeLevel0Reco_cff import *
# high level reco tasks done after Layer0 cleaners
from PhysicsTools.PatAlgos.recoLayer0.highLevelReco_cff import *
# MC matching
from PhysicsTools.PatAlgos.mcMatchLayer0.mcMatchSequences_cff import *
patLayer0Cleaners_withoutPFTau = cms.Sequence(allLayer0Muons*allLayer0Electrons*allLayer0Photons*allLayer0Jets*allLayer0METs)
patLayer0Cleaners = cms.Sequence(allLayer0Muons*allLayer0Electrons*allLayer0Photons*allLayer0Taus*allLayer0Jets*allLayer0METs)
patLayer0_withoutPFTau = cms.Sequence(patBeforeLevel0Reco_withoutPFTau*patLayer0Cleaners_withoutPFTau*patHighLevelReco_withoutPFTau*patMCTruth_withoutTau)
patLayer0 = cms.Sequence(patBeforeLevel0Reco*patLayer0Cleaners*patHighLevelReco*patMCTruth)

