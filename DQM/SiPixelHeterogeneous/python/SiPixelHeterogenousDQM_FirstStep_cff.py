import copy
import FWCore.ParameterSet.Config as cms
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorVertexSoA_cfi import *

# Run-3 sequence
monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelMonitorVertexSoA)

# Phase-2 sequence
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence(siPixelPhase2MonitorRecHitsSoA * siPixelPhase2MonitorTrackSoA * siPixelMonitorVertexSoA)
phase2_tracker.toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)

# HIon Phase 1 sequence
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

_monitorpixelSoARecHitsSourceHIon = cms.Sequence(siPixelHIonPhase1MonitorRecHitsSoA * siPixelHIonPhase1MonitorTrackSoA * siPixelMonitorVertexSoA)
pp_on_AA.toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceHIon)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareVertexSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1RawDataErrorComparator_cfi import *
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *

# digi errors
SiPixelPhase1RawDataConfForCPU = copy.deepcopy(SiPixelPhase1RawDataConf)
for pset in SiPixelPhase1RawDataConfForCPU:
    pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsCPU"

siPixelPhase1MonitorRawDataACPU = SiPixelPhase1RawDataAnalyzer.clone(
    src = "siPixelDigis@cpu",
    histograms = SiPixelPhase1RawDataConfForCPU
)

SiPixelPhase1RawDataConfForGPU = copy.deepcopy(SiPixelPhase1RawDataConf)
for pset in SiPixelPhase1RawDataConfForGPU:
    pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsGPU"

siPixelPhase1MonitorRawDataAGPU = SiPixelPhase1RawDataAnalyzer.clone(
    src = "siPixelDigis@cuda",
    histograms  =SiPixelPhase1RawDataConfForGPU
)

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

siPixelHIonPhase1MonitorRecHitsSoACPU = siPixelHIonPhase1MonitorRecHitsSoA.clone(
 pixelHitsSrc = "siPixelRecHitsPreSplittingSoA@cpu",
 TopFolderName = "SiPixelHeterogeneous/PixelRecHitsSoACPU"
)

siPixelHIonPhase1MonitorRecHitsSoAGPU = siPixelHIonPhase1MonitorRecHitsSoA.clone(
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

siPixelHIonPhase1MonitorTrackSoACPU = siPixelHIonPhase1MonitorTrackSoA.clone(
  pixelTrackSrc = 'pixelTracksSoA@cpu',
  topFolderName = 'SiPixelHeterogeneous/PixelTrackSoACPU',
)

siPixelHIonPhase1MonitorTrackSoAGPU = siPixelHIonPhase1MonitorTrackSoA.clone(
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
monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRawDataACPU *
                                            siPixelPhase1MonitorRawDataAGPU *
                                            siPixelPhase1MonitorRecHitsSoACPU *
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

# HIon sequence
_monitorpixelSoACompareSourceHIonPhase1 =  cms.Sequence(siPixelHIonPhase1MonitorRecHitsSoACPU *
                                              siPixelHIonPhase1MonitorRecHitsSoAGPU *
                                              siPixelHIonPhase1CompareRecHitsSoA *
                                              siPixelHIonPhase1MonitorTrackSoAGPU *
                                              siPixelHIonPhase1MonitorTrackSoACPU *
                                              siPixelHIonPhase1CompareTrackSoA *
                                              siPixelMonitorVertexSoACPU *
                                              siPixelMonitorVertexSoAGPU *
                                              siPixelCompareVertexSoA)

phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,_monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)
