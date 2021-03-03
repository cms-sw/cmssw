import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.hiRegitInitialStep_cff import *
from RecoHI.HiTracking.hiRegitLowPtTripletStep_cff import *
from RecoHI.HiTracking.hiRegitPixelPairStep_cff import *
from RecoHI.HiTracking.hiRegitDetachedTripletStep_cff import *
from RecoHI.HiTracking.hiRegitMixedTripletStep_cff import *

from RecoHI.HiTracking.MergeRegitTrackCollectionsHI_cff import *

hiRegitTrackingTask = cms.Task(
    hiRegitInitialStepTask
    ,hiRegitLowPtTripletStepTask
    ,hiRegitPixelPairStepTask
    ,hiRegitDetachedTripletStepTask
    ,hiRegitMixedTripletStepTask
    ,hiRegitTracks
    )
hiRegitTracking = cms.Sequence(hiRegitTrackingTask)

# Define region around jet 

hiRegitInitialStepSeeds = hiRegitInitialStepSeeds.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 0.02,
                                                originHalfLength = 0.02,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
hiRegitLowPtTripletStepSeeds = hiRegitLowPtTripletStepSeeds.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 0.02,
                                                originHalfLength = 0.02,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
hiRegitPixelPairStepSeeds = hiRegitPixelPairStepSeeds.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 0.015,
                                                originHalfLength = 0.015,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
hiRegitDetachedTripletStepSeeds = hiRegitDetachedTripletStepSeeds.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 1.5,
                                                originHalfLength = 1.5,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
hiRegitMixedTripletStepSeedsA = hiRegitMixedTripletStepSeedsA.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 0.5,
                                                originHalfLength = 0.5,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
hiRegitMixedTripletStepSeedsB = hiRegitMixedTripletStepSeedsB.clone(
    RegionFactoryPSet = dict( RegionPSet = dict(originRadius     = 0.5,
                                                originHalfLength = 0.5,
                                                deltaPhiRegion   = 0.4,
                                                deltaEtaRegion   = 0.4)
                            )
)
