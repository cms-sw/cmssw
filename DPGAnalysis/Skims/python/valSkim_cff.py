from DPGAnalysis.Skims.goodvertexSkim_cff import *

###Tracks selection
trackSelector  =cms.EDFilter("TrackSelector",
                                    src = cms.InputTag("generalTracks"),
                                     cut = cms.string('quality("highPurity")')     
                                     )

#trackSelector = cms.EDProducer("QualityFilter",
#                                       TrackQuality = cms.string('highPurity'),
#                                       recTracks = cms.InputTag("generalTracks")
#                                       )

trackFilter = cms.EDFilter("TrackCountFilter",
                                   src = cms.InputTag("trackSelector"),
                                   minNumber = cms.uint32(10)
                                   )

nottoomanytracks = cms.EDFilter("NMaxPerLumi",
                                        nMaxPerLumi = cms.uint32(8)
                                        )
relvaltrackSkim = cms.Sequence(goodvertexSkim+trackSelector + trackFilter + nottoomanytracks )

### muon selection
muonSelector = cms.EDFilter("MuonSelector",
                                    src = cms.InputTag("muons"),
                                    cut = cms.string(" isGlobalMuon && isTrackerMuon && pt > 3")
                                    )
muonFilter = cms.EDFilter("MuonCountFilter",
                                  src = cms.InputTag("muonSelector"),
                                  minNumber = cms.uint32(1)
                                  )
nottoomanymuons = cms.EDFilter("NMaxPerLumi",
                                       nMaxPerLumi = cms.uint32(2)
                                       )
relvalmuonSkim = cms.Sequence(goodvertexSkim+muonSelector + muonFilter + nottoomanymuons )

