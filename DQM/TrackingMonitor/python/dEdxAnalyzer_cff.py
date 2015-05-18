import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.dEdxAnalyzer_cfi import *

#ADD BY LOIC
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
RefitterForDedxDQMDeDx = TrackRefitter.clone()
RefitterForDedxDQMDeDx.src = cms.InputTag("generalTracks")
RefitterForDedxDQMDeDx.TrajectoryInEvent = True

from RecoTracker.DeDx.dedxEstimators_cff import dedxHarmonic2
dedxDQMHarm2SP = dedxHarmonic2.clone()
#dedxDQMHarm2SP.tracks                     = cms.InputTag("RefitterForDedxDQMDeDx")
#dedxDQMHarm2SP.trajectoryTrackAssociation = cms.InputTag("RefitterForDedxDQMDeDx")
dedxDQMHarm2SP.tracks                     = cms.InputTag("generalTracks")
dedxDQMHarm2SP.trajectoryTrackAssociation = cms.InputTag("generalTracks")
dedxDQMHarm2SP.UseStrip = cms.bool(True)
dedxDQMHarm2SP.UsePixel = cms.bool(True)

dedxDQMHarm2SO = dedxDQMHarm2SP.clone()
dedxDQMHarm2SO.UsePixel = cms.bool(False)

dedxDQMHarm2PO = dedxDQMHarm2SP.clone()
dedxDQMHarm2PO.UseStrip = cms.bool(False)


from RecoTracker.DeDx.dedxEstimators_cff import dedxHitInfo
dedxDQMHitInfo = dedxHitInfo.clone()
dedxDQMHitInfo.tracks                     = cms.InputTag("generalTracks")
dedxDQMHitInfo.trajectoryTrackAssociation = cms.InputTag("generalTracks")

#dEdxMonitor = cms.Sequence(
#    RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO
#    * dEdxAnalyzer
#)
