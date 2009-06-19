import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfTopProjectionPFTausOnPFJets_cfi import pfTopProjectionPFTausOnPFJets as tp
tausOnJets = tp.clone()
tausOnJets.name = 'tausOnJets'
tausOnJets.topCollection = 'allLayer0Taus'
tausOnJets.bottomCollection = 'pfJets'

dump = cms.EDAnalyzer("EventContentAnalyzer")

pfJetsNoTauSequence = cms.Sequence(
    tausOnJets
    )

