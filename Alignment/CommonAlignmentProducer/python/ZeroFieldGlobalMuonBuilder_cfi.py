ZeroFieldGlobalMuonBuilder = cms.EDFilter("ZeroFieldGlobalMuonBuilder",
                                          inputTracker = cms.InputTag("cosmictrackfinderP5"),
                                          inputMuon = cms.InputTag("cosmicMuons"),
                                          minTrackerHits = cms.int32(0),
                                          minMuonHits = cms.int32(0),
                                          minPdot = cms.double(0.99),
                                          minDdotP = cms.double(0.99),
                                          debuggingHistograms = cms.untracked.bool(False),
                                          )
