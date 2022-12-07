import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorRecHitsSoAPhase1_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorRecHitsSoAPhase2_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoAPhase1_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoAPhase2_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorVertexSoA_cfi import *

monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoAPhase1 * siPixelPhase1MonitorTrackSoAPhase1 * siPixelPhase1MonitorVertexSoA)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence(siPixelPhase1MonitorRecHitsSoAPhase2 * siPixelPhase1MonitorTrackSoAPhase2 * siPixelPhase1MonitorVertexSoA)
phase2_tracker.toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareRecHitsSoAPhase1_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareRecHitsSoAPhase2_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareTrackSoAPhase1_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareTrackSoAPhase2_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareVertexSoA_cfi import *

siPixelPhase1MonitorRecHitsSoACPU = siPixelPhase1MonitorRecHitsSoAPhase1.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelPhase1MonitorRecHitsSoAGPU = siPixelPhase1MonitorRecHitsSoAPhase1.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cuda",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoAGPU"
)

siPixelPhase2MonitorRecHitsSoACPU = siPixelPhase1MonitorRecHitsSoAPhase2.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelPhase2MonitorRecHitsSoAGPU = siPixelPhase1MonitorRecHitsSoAPhase2.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cuda",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoAGPU"
)

siPixelPhase1MonitorTrackSoACPU = siPixelPhase1MonitorTrackSoAPhase1.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelPhase1MonitorTrackSoAGPU = siPixelPhase1MonitorTrackSoAPhase1.clone(
  pixelTrackSrc = 'pixelTracksSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoAGPU',
)

siPixelPhase2MonitorTrackSoACPU = siPixelPhase1MonitorTrackSoAPhase2.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelPhase2MonitorTrackSoAGPU = siPixelPhase1MonitorTrackSoAPhase2.clone(
  pixelTrackSrc = 'pixelTracksSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoAGPU',
)

siPixelMonitorVertexSoACPU = siPixelPhase1MonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoACPU',
)

siPixelMonitorVertexSoAGPU = siPixelPhase1MonitorVertexSoA.clone(
  pixelVertexSrc = 'pixelVerticesSoA@cuda',
  topFolderName = 'SiPixelHeterogeneous/PixelVertexSoAGPU',
)

monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRecHitsSoACPU *
                                            siPixelPhase1MonitorRecHitsSoAGPU *
                                            siPixelPhase1CompareRecHitsSoAPhase1 *
                                            siPixelPhase1MonitorTrackSoAGPU *
                                            siPixelPhase1MonitorTrackSoACPU *
                                            siPixelPhase1CompareTrackSoAPhase1 *
                                            siPixelMonitorVertexSoACPU *
                                            siPixelMonitorVertexSoAGPU *
                                            siPixelPhase1CompareVertexSoA)

_monitorpixelSoACompareSource =  cms.Sequence(siPixelPhase2MonitorRecHitsSoACPU *
                                              siPixelPhase2MonitorRecHitsSoAGPU *
                                              siPixelPhase1CompareRecHitsSoAPhase2 *
                                              siPixelPhase2MonitorTrackSoAGPU *
                                              siPixelPhase2MonitorTrackSoACPU *
                                              siPixelPhase1CompareTrackSoAPhase2 *
                                              siPixelMonitorVertexSoACPU *
                                              siPixelMonitorVertexSoAGPU *
                                              siPixelPhase1CompareVertexSoA)

phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,_monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)
