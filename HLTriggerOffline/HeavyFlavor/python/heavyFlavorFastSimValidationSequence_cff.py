from HLTriggerOffline.HeavyFlavor.heavyFlavorValidation_cfi import *
heavyFlavorValidation.L1Muons = cms.InputTag("l1extraParticles","","HLT")
heavyFlavorValidation.RecoMuons = cms.InputTag("muons","","HLT")
heavyFlavorValidation.GlobMuons = cms.InputTag("globalMuons","","HLT")

heavyFlavorFastSimValidationSequence = cms.Sequence(heavyFlavorValidation)
