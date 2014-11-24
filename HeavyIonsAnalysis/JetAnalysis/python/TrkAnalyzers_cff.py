import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.TrackAnalysis.trackAnalyzer_cff import *

anaTrack.trackPtMin = 0.4
anaTrack.useQuality = False
anaTrack.doPFMatching = True
anaTrack.pfCandSrc = cms.InputTag("particleFlowTmp")
anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")

anaTrack.qualityStrings = cms.untracked.vstring('highPurity','highPuritySetWithPV')

pixelTrack = anaTrack.clone(trackSrc = cms.InputTag("hiPixel3PrimTracks"))
pixelTrack.useQuality = False
pixelTrack.trackPtMin = 0.4

mergedTrack = pixelTrack.clone(trackSrc = cms.InputTag("hiMergedTracks"))

ppTrack = anaTrack.clone(trackSrc = cms.InputTag("generalTracks"),
                         vertexSrc = ["offlinePrimaryVerticesWithBS"]
                         )

