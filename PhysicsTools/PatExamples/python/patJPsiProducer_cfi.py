import FWCore.ParameterSet.Config as cms

patJPsiCandidates = cms.EDFilter("PatJPsiProducer",
                                 muonSrc = cms.InputTag("selectedLayer1Muons")
                                 )
