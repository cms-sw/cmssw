import FWCore.ParameterSet.Config as cms

from ..modules.generalTracks_cfi import *
from ..modules.hltPhase2PixelVertices_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..sequences.highPtTripletStepSequence_cfi import *
from ..sequences.hltPhase2PixelTracksSequence_cfi import *
from ..sequences.initialStepSequence_cfi import *
from ..sequences.itLocalRecoSequence_cfi import *
from ..sequences.otLocalRecoSequence_cfi import *

HLTTrackingV61Sequence = cms.Sequence((itLocalRecoSequence+otLocalRecoSequence+trackerClusterCheck+hltPhase2PixelTracksSequence+hltPhase2PixelVertices+initialStepSequence+highPtTripletStepSequence+generalTracks))
