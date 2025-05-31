import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates_default_cfi import _templates_default
templates = _templates_default.clone()

# Enable the good edge algorithm in pixel hit reconstruction that handles broken/truncated pixel cluster caused by radiation damage
from Configuration.ProcessModifiers.siPixelGoodEdgeAlgo_cff import siPixelGoodEdgeAlgo
siPixelGoodEdgeAlgo.toModify(templates, GoodEdgeAlgo = True)
