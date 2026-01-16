import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates2_default_cfi import _templates2_default
templates2 = _templates2_default.clone()
templates2_speed0 = _templates2_default.clone(
    ComponentName = "PixelCPEClusterRepairWithoutProbQ",
    speed = 0
)

# Enable the good edge algorithm in pixel hit reconstruction that handles broken/truncated pixel cluster caused by radiation damage
from Configuration.ProcessModifiers.siPixelGoodEdgeAlgo_cff import siPixelGoodEdgeAlgo
siPixelGoodEdgeAlgo.toModify(templates2, GoodEdgeAlgo = True)
siPixelGoodEdgeAlgo.toModify(templates2_speed0, GoodEdgeAlgo = True)
