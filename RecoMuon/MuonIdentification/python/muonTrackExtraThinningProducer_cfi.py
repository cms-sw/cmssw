import FWCore.ParameterSet.Config as cms

muonTrackExtraThinningProducer = cms.EDProducer("MuonTrackExtraThinningProducer",
                                                inputTag = cms.InputTag("generalTracks"),
                                                cut = cms.string("pt > 3. || isPFMuon"),
                                                slimTrajParams = cms.bool(True),
                                                slimResiduals = cms.bool(True),
                                                slimFinalState = cms.bool(True),
                                                muonTag = cms.InputTag("muons"),
                                                )
