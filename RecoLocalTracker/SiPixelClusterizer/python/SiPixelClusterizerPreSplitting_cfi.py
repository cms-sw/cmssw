import FWCore.ParameterSet.Config as cms

from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
siPixelClustersPreSplitting = _siPixelClusters.clone()

# In principle we could remove `siPixelClustersPreSplitting` from the `pixeltrackerlocalreco` 
# sequence when the `gpu` modufier is active; for the time being we keep it for simplicity.
