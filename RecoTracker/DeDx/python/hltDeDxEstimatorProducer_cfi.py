import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxEstimators_cff import *

DeDxEstimatorProducer = dedxHarmonic2.clone(tracks = "hltIter4Merged")
