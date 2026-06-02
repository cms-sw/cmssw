import FWCore.ParameterSet.Config as cms

from .RecoTrackFromScoutingMonitor_cfi import scoutingRecoTrackMonitor

# Scouting Tracks to reco::Track conversion
recoTracksFromScouting = cms.EDProducer("Run3ScoutingTrackToRecoTrackProducer",
                                        skipMissingProduct = cms.bool(True), # do not throw on missing input
					src = cms.InputTag("hltScoutingTrackPacker"))

# Scouting Vertices to reco::Vertex conversion
recoVerticesFromScouting = cms.EDProducer("Run3ScoutingVertexToRecoVertexProducer",
                                          skipMissingProduct = cms.bool(True), # do not throw on missing input
					  src = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"))

recoTrackFromScoutingMonitorSequence = cms.Sequence(recoTracksFromScouting +
                                                    recoVerticesFromScouting +
                                                    scoutingRecoTrackMonitor)
