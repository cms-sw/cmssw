import FWCore.ParameterSet.Config as cms

from .RecoTrackFromScoutingMonitor_cfi import scoutingRecoTrackMonitor

# Tracks
recoTracksFromScouting = cms.EDProducer("Run3ScoutingTrackToRecoTrackProducer",
					src = cms.InputTag("hltScoutingTrackPacker"))

# Vertices 
recoVerticesFromScouting = cms.EDProducer("Run3ScoutingVertexToRecoVertexProducer",
					  src = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"))

recoTrackFromScoutingMonitorSequence = cms.Sequence(recoTracksFromScouting +
                                                    recoVerticesFromScouting +
                                                    scoutingRecoTrackMonitor)
