import FWCore.ParameterSet.Config as cms

# SiPixelGainCalibrationServiceParameters
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *

# legacy pixel cluster producer
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
siPixelClustersPreSplitting = _siPixelClusters.clone() 

from Configuration.ProcessModifiers.siPixelDigiMorphing_cff import siPixelDigiMorphing
siPixelDigiMorphing.toModify(siPixelClustersPreSplitting,
                             src = 'siPixelDigisMorphed'
                             )
