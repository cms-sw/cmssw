import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.TrackAnalysis.unpackedTracksAndVertices_cfi import *
from HeavyIonsAnalysis.TrackAnalysis.TrackAnalyzer_cfi import *

PbPbTracks = trackAnalyzer.clone()

ppTracks = trackAnalyzer.clone()

trackSequencePbPb = cms.Sequence(unpackedTracksAndVertices + PbPbTracks)

trackSequencePP = cms.Sequence(unpackedTracksAndVertices + ppTracks)
