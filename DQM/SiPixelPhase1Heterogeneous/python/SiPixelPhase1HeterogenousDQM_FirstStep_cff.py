import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorVertexSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(siPixelPhase1MonitorRecHitsSoA, pixelHitsSrc = "siPixelRecHitsPreSplittingSoA")


monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelPhase1MonitorVertexSoA)
