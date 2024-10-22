import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates2_default_cfi import _templates2_default
templates2 = _templates2_default.clone()
templates2_speed0 = _templates2_default.clone(
    ComponentName = "PixelCPEClusterRepairWithoutProbQ",
    speed = 0
)
