
import FWCore.ParameterSet.Config as cms

#
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
from RecoLocalTracker.Phase2ITPixelClusterizer.Phase2ITPixelClusterizer_cfi import phase2ITPixelClusters as _phase2ITPixelClusters
phase2ITPixelClustersPreSplitting = _phase2ITPixelClusters.clone()
