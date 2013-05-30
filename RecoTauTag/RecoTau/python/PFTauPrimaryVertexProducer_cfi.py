import FWCore.ParameterSet.Config as cms

PFTauPrimaryVertexProducer = cms.EDProducer("PFTauPrimaryVertexProducer",
                                            PFTauTag =  cms.InputTag("hpsPFTauProducer"),
                                            ElectronTag = cms.InputTag("MyElectrons"),
                                            MuonTag = cms.InputTag("MyMuons"),
                                            PVTag = cms.InputTag("offlinePrimaryVertices"),
                                            beamSpot = cms.InputTag("offlineBeamSpot"),
                                            TrackCollectionTag = cms.InputTag("generalTracks"),
                                            Algorithm = cms.int32(1),
                                            useBeamSpot = cms.bool(True),
                                            RemoveMuonTracks = cms.bool(False),
                                            RemoveElectronTracks = cms.bool(False),
                                            useSelectedTaus = cms.bool(False)
                                            )

