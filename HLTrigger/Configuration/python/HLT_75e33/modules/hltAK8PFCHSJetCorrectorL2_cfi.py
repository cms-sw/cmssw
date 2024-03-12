import FWCore.ParameterSet.Config as cms

hltAK8PFCHSJetCorrectorL2 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L2Relative')
)
# foo bar baz
# P1Q4O0tu9pqcM
