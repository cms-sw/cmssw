import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.dEdxAnalyzer_cfi import *

#ADD BY LOIC
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
RefitterForDedxDQMDeDx = TrackRefitter.clone(
    src = "generalTracks",
    TrajectoryInEvent = True
)

from RecoTracker.DeDx.dedxEstimators_cff import dedxHarmonic2
dedxDQMHarm2SP = dedxHarmonic2.clone(
    #tracks = "RefitterForDedxDQMDeDx",
    tracks = "generalTracks",
    UseStrip = True,
    UsePixel = True
)

dedxDQMHarm2SO = dedxDQMHarm2SP.clone(
    UsePixel = False
)

dedxDQMHarm2PO = dedxDQMHarm2SP.clone(
    UseStrip = False
)

#dEdxMonitor = cms.Sequence(
#    RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO
#    * dEdxAnalyzer
#)
