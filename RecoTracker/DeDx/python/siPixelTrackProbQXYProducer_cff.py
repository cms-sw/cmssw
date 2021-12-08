import FWCore.ParameterSet.Config as cms
import RecoTracker.DeDx.siPixelTrackProbQXYProducer_cfi as _probQXY

siPixelTrackProbQXY = _probQXY.siPixelTrackProbQXYProducer.clone()

doSiPixelTrackProbQXYTask = cms.Task(siPixelTrackProbQXY)
doSiPixelTrackProbQXY = cms.Sequence(doSiPixelTrackProbQXYTask)
