import FWCore.ParameterSet.Config as cms

import copy
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
#   request Pixel track finding with very low pt threshold. Even if only high-pt tracks
#   are selected, the low-Pt might be wanted to check isolation of the high-Pt track.
#   otherwise the ptMin here can be increased.
pixelTracksForMinBias = copy.deepcopy(pixelTracks)
#include "RecoLocalTracker/Configuration/data/RecoLocalTracker.cff"
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
import copy
from HLTrigger.special.TriggerTypeFilter_cfi import *
#
# TriggerType filter:
#
filterTriggerType = copy.deepcopy(triggerTypeFilter)
#    alternative to the above. Seems to work (and be a bit more efficient due to ptMin=0.2->0.075), but not properly tested yet.
#    module pixelTracksForMinBias = pixelLowPtTracksWithZPos from "RecoPixelVertexing/PixelLowPtUtilities/data/PixelLowPtTracksWithZPos.cfi"
pixelTrackingForMinBias = cms.Sequence(pixelTracksForMinBias)
pixelTrackingForIsol = cms.Sequence(pixelTracks)
pixelTracksForMinBias.RegionFactoryPSet.RegionPSet.ptMin = 0.2
filterTriggerType.InputLabel = 'rawDataCollector'

