import FWCore.ParameterSet.Config as cms
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorVertexSoA_cfi import *

# Run-3 sequence
monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelMonitorVertexSoA)

# Phase-2 sequence
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence(siPixelPhase2MonitorRecHitsSoA * siPixelPhase2MonitorTrackSoA * siPixelMonitorVertexSoA)
phase2_tracker.toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareVertexSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1RawDataErrorComparator_cfi import *

## rechits
siPixelPhase1MonitorRecHitsSoACPU = siPixelPhase1MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelPhase1MonitorRecHitsSoAGPU = siPixelPhase1MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cuda",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoAGPU"
)

siPixelPhase2MonitorRecHitsSoACPU = siPixelPhase2MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelPhase2MonitorRecHitsSoAGPU = siPixelPhase2MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cuda",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoAGPU"
)

## tracks
siPixelPhase1MonitorTrackSoACPU = siPixelPhase1MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelPhase1MonitorTrackSoAGPU = siPixelPhase1MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoAGPU',
)

siPixelPhase2MonitorTrackSoACPU = siPixelPhase2MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelPhase2MonitorTrackSoAGPU = siPixelPhase2MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoAGPU',
)

## vertices
siPixelMonitorVertexSoACPU = siPixelMonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoACPU',
)

siPixelMonitorVertexSoAGPU = siPixelMonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoAGPU',
)

# Run-3 sequence
monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRecHitsSoACPU *
                                            siPixelPhase1MonitorRecHitsSoAGPU *
                                            siPixelPhase1CompareRecHitsSoA *
                                            siPixelPhase1MonitorTrackSoAGPU *
                                            siPixelPhase1MonitorTrackSoACPU *
                                            siPixelPhase1CompareTrackSoA *
                                            siPixelMonitorVertexSoACPU *
                                            siPixelMonitorVertexSoAGPU *
                                            siPixelCompareVertexSoA *
                                            siPixelPhase1RawDataErrorComparator)

# Phase-2 sequence
_monitorpixelSoACompareSource =  cms.Sequence(siPixelPhase2MonitorRecHitsSoACPU *
                                              siPixelPhase2MonitorRecHitsSoAGPU *
                                              siPixelPhase2CompareRecHitsSoA *
                                              siPixelPhase2MonitorTrackSoAGPU *
                                              siPixelPhase2MonitorTrackSoACPU *
                                              siPixelPhase2CompareTrackSoA *
                                              siPixelMonitorVertexSoACPU *
                                              siPixelMonitorVertexSoAGPU *
                                              siPixelCompareVertexSoA)

phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,_monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)
