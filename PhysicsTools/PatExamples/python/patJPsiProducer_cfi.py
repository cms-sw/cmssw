import FWCore.ParameterSet.Config as cms

patJPsiCandidates = cms.EDProducer("PatJPsiProducer",
                                 muonSrc = cms.InputTag("selectedLayer1Muons")
                                 )
# foo bar baz
# 28KQShPzIRi4O
