import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.TrackAnalysis.trackAnalyzer_cff import *

anaTrack.trackPtMin = 0.49
anaTrack.useQuality = False
anaTrack.doPFMatching = True
anaTrack.doSimVertex = False  
anaTrack.doSimTrack = False
anaTrack.pfCandSrc = cms.InputTag("particleFlowTmp")
anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")
anaTrack.qualityStrings = cms.untracked.vstring(['highPurity','tight','loose'])

pixelTrack = anaTrack.clone(trackSrc = cms.InputTag("hiGeneralAndPixelTracks"),
                            useQuality = False,
							trackPtMin = 0.4,
							qualityStrings = cms.untracked.vstring('highPurity'))

ppTrack = anaTrack.clone(trackSrc = cms.InputTag("generalTracks"),
                         vertexSrc = ["offlinePrimaryVertices"],
                         mvaSrc = cms.InputTag("generalTracks","MVAVals"),
						 qualityStrings = cms.untracked.vstring(['highPurity','tight','loose']),
						 doPFMatching = True,
						 pfCandSrc = cms.InputTag('particleFlow'),
                         doSimVertex = False,  
                         doSimTrack = False
                         )

trackSequencesPbPb = cms.Sequence(anaTrack)
								  
trackSequencesPP = cms.Sequence(ppTrack)
							
