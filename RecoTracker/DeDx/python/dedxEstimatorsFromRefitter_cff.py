import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.TrackRefitter_cfi import *

RefitterForDeDx = TrackRefitter.clone(
    TrajectoryInEvent = True
)

from RecoTracker.DeDx.dedxEstimators_cff import *

dedxHitInfo.tracks="RefitterForDeDx"
dedxHitInfo.trajectoryTrackAssociation = "RefitterForDeDx"

dedxHarmonic2.tracks="RefitterForDeDx"
dedxHarmonic2.trajectoryTrackAssociation = "RefitterForDeDx"

dedxTruncated40.tracks="RefitterForDeDx"
dedxTruncated40.trajectoryTrackAssociation = "RefitterForDeDx"

dedxMedian.tracks="RefitterForDeDx"
dedxMedian.trajectoryTrackAssociation = "RefitterForDeDx"

dedxUnbinned.tracks="RefitterForDeDx"
dedxUnbinned.trajectoryTrackAssociation = "RefitterForDeDx"

dedxDiscrimProd.tracks="RefitterForDeDx"
dedxDiscrimProd.trajectoryTrackAssociation = "RefitterForDeDx"

dedxDiscrimBTag.tracks="RefitterForDeDx"
dedxDiscrimBTag.trajectoryTrackAssociation = "RefitterForDeDx"

dedxDiscrimSmi.tracks="RefitterForDeDx"
dedxDiscrimSmi.trajectoryTrackAssociation = "RefitterForDeDx"

dedxDiscrimASmi.tracks="RefitterForDeDx"
dedxDiscrimASmi.trajectoryTrackAssociation = "RefitterForDeDx"

doAlldEdXEstimatorsTask = cms.Task(RefitterForDeDx, dedxTruncated40, dedxHarmonic2, dedxHitInfo )
doAlldEdXEstimators = cms.Sequence(doAlldEdXEstimatorsTask)
