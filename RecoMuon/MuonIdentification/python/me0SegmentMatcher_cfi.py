import FWCore.ParameterSet.Config as cms


me0SegmentMatching = cms.EDProducer("ME0SegmentMatcher",
                                    maxPullX = cms.double (3.0),
                                    maxDiffX = cms.double (4.0),
                                    maxPullY = cms.double (20.0),
                                    maxDiffY = cms.double (20.0),
                                    maxDiffPhiDirection = cms.double (3.14),
                                    me0SegmentTag = cms.InputTag("me0Segments"),
                                    tracksTag = cms.InputTag("generalTracks")
                                    )
