import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxEstimators_cff import *

DeDxEstimatorProducer = dedxHarmonic2.clone()
DeDxEstimatorProducer.tracks=cms.InputTag("hltIter4Merged")
DeDxEstimatorProducer.trajectoryTrackAssociation = cms.InputTag("hltIter4Merged")
