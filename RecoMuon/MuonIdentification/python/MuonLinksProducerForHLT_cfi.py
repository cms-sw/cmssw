import FWCore.ParameterSet.Config as cms

hltMuonLinks = cms.EDProducer("MuonLinksProducerForHLT",
                              InclusiveTrackerTrackCollection = cms.InputTag("hltPFMuonMerging"),
                              LinkCollection = cms.InputTag("hltL3MuonsLinksCombination"),
                              ptMin = cms.double(2.5),
                              pMin = cms.double(2.5),
                              shareHitFraction = cms.double(0.80)
                              )
