import FWCore.ParameterSet.Config as cms


from RecoTracker.TrackProducer.TrackRefitter_cfi import *
RefitterForDeDxDiscrim = TrackRefitter.clone()
RefitterForDeDxDiscrim.TrajectoryInEvent = True

from RecoTracker.DeDx.dedxDiscriminators_cff import *
dedxDiscrimProdFromReffiter         = dedxDiscrimProd.clone()
dedxDiscrimProdFromReffiter.tracks  = cms.InputTag("RefitterForDeDxDiscrim")
dedxDiscrimProdFromReffiter.trajectoryTrackAssociation = cms.InputTag("RefitterForDeDxDiscrim")

dedxDiscrimBTagFromReffiter         = dedxDiscrimProdFromReffiter.clone()
dedxDiscrimBTagFromReffiter.Formula = cms.untracked.uint32(1)

dedxDiscrimSmiFromReffiter         = dedxDiscrimProdFromReffiter.clone()
dedxDiscrimSmiFromReffiter.Formula = cms.untracked.uint32(2)

dedxDiscrimASmiFromReffiter         = dedxDiscrimProdFromReffiter.clone()
dedxDiscrimASmiFromReffiter.Formula = cms.untracked.uint32(3)

doAlldEdXDiscriminatorsFromReffiter = cms.Sequence(dedxDiscrimProdFromReffiter * dedxDiscrimBTagFromReffiter * dedxDiscrimSmiFromReffiter * dedxDiscrimASmiFromReffiter)

