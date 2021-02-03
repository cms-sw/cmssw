import FWCore.ParameterSet.Config as cms

from ..modules.generalTracks_cfi import *
from ..modules.offlineBeamSpot_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..sequences.highPtTripletStepSequence_cfi import *
from ..sequences.initialStepSequence_cfi import *
from ..sequences.itLocalReco_cfi import *
from ..sequences.otLocalReco_cfi import *
from ..sequences.pixelTracksSequence_cfi import *
from ..sequences.pixelVerticesSequence_cfi import *
from ..sequences.vertexReco_cfi import *

globalreco_tracking = cms.Sequence(offlineBeamSpot+itLocalReco+otLocalReco+trackerClusterCheck+pixelTracksSequence+pixelVerticesSequence+initialStepSequence+highPtTripletStepSequence+generalTracks+vertexReco)
