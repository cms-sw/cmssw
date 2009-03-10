import FWCore.ParameterSet.Config as cms

# high level reco tasks needed before Layer0 cleaners
from PhysicsTools.PatAlgos.recoLayer0.beforeLevel0Reco_cff import *

# full cleaning
from PhysicsTools.PatAlgos.cleaningLayer0.caloJetCleaner_cfi  import *
from PhysicsTools.PatAlgos.cleaningLayer0.pfJetCleaner_cfi    import * ## but not in the sequence by default
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi  import *
from PhysicsTools.PatAlgos.cleaningLayer0.electronCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi     import *
from PhysicsTools.PatAlgos.cleaningLayer0.pfTauCleaner_cfi    import *
from PhysicsTools.PatAlgos.cleaningLayer0.photonCleaner_cfi   import *

patLayer0Cleaners_withoutPFTau = cms.Sequence(
        allLayer0Muons *
        allLayer0Electrons *
        allLayer0Photons *
        allLayer0Jets *
        allLayer0METs
)

patLayer0Cleaners = cms.Sequence(
        allLayer0Muons *
        allLayer0Electrons *
        allLayer0Photons *
        allLayer0Taus *
        allLayer0Jets *
        allLayer0METs
)

# high level reco tasks done after Layer0 cleaners
from PhysicsTools.PatAlgos.recoLayer0.highLevelReco_cff    import  *

# MC matching
from PhysicsTools.PatAlgos.mcMatchLayer0.mcMatchSequences_cff   import  *

# trigger matching
from PhysicsTools.PatAlgos.triggerLayer0.trigMatchSequences_cff import  *


patLayer0_withoutPFTau_withoutTrigMatch = cms.Sequence(
        patBeforeLevel0Reco_withoutPFTau *
        patLayer0Cleaners_withoutPFTau *
        patHighLevelReco_withoutPFTau *
        patMCTruth_withoutTau
)
patLayer0_withoutTrigMatch = cms.Sequence(
        patBeforeLevel0Reco *
        patLayer0Cleaners *
        patHighLevelReco *
        patMCTruth
)

patLayer0_withoutPFTau = cms.Sequence(
        patLayer0_withoutPFTau_withoutTrigMatch *
        patTrigMatch_withoutBTau
)

# Layer 0 default sequence 
patLayer0 = cms.Sequence(
        patLayer0_withoutTrigMatch *
        patTrigMatch
)

patLayer0_patTuple_withoutPFTau = cms.Sequence(
        patLayer0_withoutPFTau_withoutTrigMatch *
        patTrigMatch_patTuple_withoutBTau
)

patLayer0_patTuple = cms.Sequence(
        patLayer0_withoutTrigMatch *
        patTrigMatch_patTuple
)
