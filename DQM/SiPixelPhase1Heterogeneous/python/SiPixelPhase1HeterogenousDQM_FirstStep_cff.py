import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorVertexSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *

monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelPhase1MonitorVertexSoA)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA)
phase2_tracker.toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareVertexSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareRecHitsSoA_cfi import *

siPixelPhase1MonitorTrackSoACPU = siPixelPhase1MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelPhase1MonitorTrackSoAGPU = siPixelPhase1MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoAGPU',
)

siPixelPhase1MonitorVertexSoACPU = siPixelPhase1MonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoACPU',
)

siPixelPhase1MonitorVertexSoAGPU = siPixelPhase1MonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoAGPU',
)

siPixelPhase1MonitorRecHitsSoACPU = siPixelPhase1MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelPhase1MonitorRecHitsSoAGPU = siPixelPhase1MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cuda",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoAGPU"
)

monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRecHitsSoACPU *
                                            siPixelPhase1MonitorRecHitsSoAGPU *
                                            siPixelPhase1CompareRecHitsSoA *
                                            siPixelPhase1MonitorTrackSoAGPU *
                                            siPixelPhase1MonitorTrackSoACPU *
                                            siPixelPhase1CompareTrackSoA *
                                            siPixelPhase1MonitorVertexSoACPU *
                                            siPixelPhase1MonitorVertexSoAGPU *
                                            siPixelPhase1CompareVertexSoA)

monitorpixelSoACompareSourceNoTracking = cms.Sequence(siPixelPhase1MonitorRecHitsSoACPU *
                                      	              siPixelPhase1MonitorRecHitsSoAGPU *
                                        	      siPixelPhase1CompareRecHitsSoA)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,monitorpixelSoACompareSourceNoTracking)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)
