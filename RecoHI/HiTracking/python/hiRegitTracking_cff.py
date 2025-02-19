import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.hiRegitInitialStep_cff import *
from RecoHI.HiTracking.hiRegitLowPtTripletStep_cff import *
from RecoHI.HiTracking.hiRegitPixelPairStep_cff import *
from RecoHI.HiTracking.hiRegitDetachedTripletStep_cff import *
from RecoHI.HiTracking.hiRegitMixedTripletStep_cff import *

from RecoHI.HiTracking.MergeRegitTrackCollectionsHI_cff import *

hiRegitTracking = cms.Sequence(
    hiRegitInitialStep
    *hiRegitLowPtTripletStep
    *hiRegitPixelPairStep
    *hiRegitDetachedTripletStep
    *hiRegitMixedTripletStep
    *hiRegitTracks
    )





