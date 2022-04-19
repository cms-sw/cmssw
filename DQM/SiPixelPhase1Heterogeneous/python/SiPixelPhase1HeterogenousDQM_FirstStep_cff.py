import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorVertexSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(siPixelPhase1MonitorRecHitsSoA, pixelHitsSrc = "siPixelRecHitsPreSplittingSoA")


monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelPhase1MonitorVertexSoA)


#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1CompareVertexSoA_cfi import *

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


monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorTrackSoAGPU *
                                            siPixelPhase1MonitorTrackSoACPU *
                                            siPixelPhase1CompareTrackSoA *
                                            siPixelPhase1MonitorVertexSoACPU *
                                            siPixelPhase1MonitorVertexSoAGPU *
                                            siPixelPhase1CompareVertexSoA
)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)
