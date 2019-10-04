import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.TrackRefitter_cfi import *

RefitterForDeDx = TrackRefitter.clone()
RefitterForDeDx.TrajectoryInEvent = True

from RecoTracker.DeDx.dedxEstimators_cff import *

dedxHitInfo.tracks=cms.InputTag("RefitterForDeDx")
dedxHitInfo.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxHarmonic2.tracks=cms.InputTag("RefitterForDeDx")
dedxHarmonic2.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxTruncated40.tracks=cms.InputTag("RefitterForDeDx")
dedxTruncated40.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxMedian.tracks=cms.InputTag("RefitterForDeDx")
dedxMedian.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxUnbinned.tracks=cms.InputTag("RefitterForDeDx")
dedxUnbinned.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxDiscrimProd.tracks=cms.InputTag("RefitterForDeDx")
dedxDiscrimProd.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxDiscrimBTag.tracks=cms.InputTag("RefitterForDeDx")
dedxDiscrimBTag.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxDiscrimSmi.tracks=cms.InputTag("RefitterForDeDx")
dedxDiscrimSmi.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

dedxDiscrimASmi.tracks=cms.InputTag("RefitterForDeDx")
dedxDiscrimASmi.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDx")

doAlldEdXEstimatorsTask = cms.Task(RefitterForDeDx, dedxTruncated40, dedxHarmonic2, dedxHitInfo )
doAlldEdXEstimators = cms.Sequence(doAlldEdXEstimatorsTask)
