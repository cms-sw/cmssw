import FWCore.ParameterSet.Config as cms

anaTrack = cms.EDAnalyzer('TrackAnalyzer',
                          trackPtMin = cms.untracked.double(0.4),
                          simTrackPtMin = cms.untracked.double(0.4),
                          vertexSrc = cms.vstring('hiSelectedVertex'),
                          trackSrc = cms.InputTag('hiGeneralTracks'),
                          mvaSrc = cms.InputTag('hiGeneralTracks','MVAVals'),
                          particleSrc = cms.InputTag('genParticles'),
                          pfCandSrc = cms.InputTag('particleFlow'),
			  beamSpotSrc = cms.untracked.InputTag('offlineBeamSpot'),
                          doPFMatching = cms.untracked.bool(False),
                          doSimTrack = cms.untracked.bool(False), 
                          doSimVertex = cms.untracked.bool(False),                          
                          useQuality = cms.untracked.bool(False),
                          qualityString = cms.untracked.string('highPurity'),
                          tpFakeSrc = cms.untracked.InputTag('mix','MergedTrackTruth'),
                          tpEffSrc = cms.untracked.InputTag('mix','MergedTrackTruth'),
                          # associateChi2 = cms.bool(False),
                          associatorMap = cms.InputTag('tpRecoAssocHiGeneralTracks'),
                          doMVA = cms.untracked.bool(True),
						  fillSimTrack = cms.untracked.bool(False)
                          )
