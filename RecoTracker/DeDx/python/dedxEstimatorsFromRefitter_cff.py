import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.TrackRefitter_cfi import *

RefitterForDeDx = TrackRefitter.clone()
RefitterForDeDx.TrajectoryInEvent = True

from RecoTracker.DeDx.dedxEstimators_cff import *
dedxTruncated40.tracks=cms.InputTag("RefitterForDeDx")
dedxTruncated40.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")
dedxHarmonic2.tracks=cms.InputTag("RefitterForDeDx")
dedxHarmonic2.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")
dedxMedian.tracks=cms.InputTag("RefitterForDeDx")
dedxMedian.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")
dedxUnbinned.tracks=cms.InputTag("RefitterForDeDx")
dedxUnbinned.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

doAlldEdXEstimators = cms.Sequence(RefitterForDeDx * (dedxTruncated40 + dedxMedian + dedxHarmonic2 + dedxUnbinned))





