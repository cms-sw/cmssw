import FWCore.ParameterSet.Config as cms

muonTrackExtraThinningProducer = cms.EDProducer("MuonTrackExtraThinningProducer",
                                                inputTag = cms.InputTag("generalTracks"),
                                                cut = cms.string("pt > 3. || isPFMuon"),
                                                muonTag = cms.InputTag("muons"),
                                                )
