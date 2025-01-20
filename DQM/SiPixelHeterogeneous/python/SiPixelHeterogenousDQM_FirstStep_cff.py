import copy
import FWCore.ParameterSet.Config as cms
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorVertexSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1StripMonitorTrackSoA_cfi import *
# Alpaka Modules
from Configuration.ProcessModifiers.alpaka_cff import alpaka
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorRecHitsSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorRecHitsSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorRecHitsSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1MonitorTrackSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2MonitorTrackSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1MonitorTrackSoAAlpaka_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorVertexSoAAlpaka_cfi import *



# Run-3 sequence
monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorRecHitsSoA * siPixelPhase1MonitorTrackSoA * siPixelMonitorVertexSoA)
# Run-3 Alpaka sequence 
monitorpixelSoASourceAlpaka = cms.Sequence(siPixelPhase1MonitorRecHitsSoAAlpaka * siPixelPhase1MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
alpaka.toReplaceWith(monitorpixelSoASource, monitorpixelSoASourceAlpaka)
# Phase-2 sequence
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence(siPixelPhase2MonitorRecHitsSoA * siPixelPhase2MonitorTrackSoA * siPixelMonitorVertexSoA)
(phase2_tracker & ~alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)
_monitorpixelSoARecHitsSourceAlpaka = cms.Sequence(siPixelPhase2MonitorRecHitsSoAAlpaka * siPixelPhase2MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
(phase2_tracker & alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceAlpaka)

# HIon Phase 1 sequence
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

_monitorpixelSoARecHitsSourceHIon = cms.Sequence(siPixelHIonPhase1MonitorRecHitsSoA * siPixelHIonPhase1MonitorTrackSoA * siPixelMonitorVertexSoA)
(pp_on_AA & ~phase2_tracker).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceHIon)
_monitorpixelSoARecHitsSourceHIonAlpaka = cms.Sequence(siPixelHIonPhase1MonitorRecHitsSoAAlpaka * siPixelHIonPhase1MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
(pp_on_AA & ~phase2_tracker & alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceHIonAlpaka)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1StripCompareTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareVertexSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1RawDataErrorComparator_cfi import *
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
#Alpaka
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareRecHits_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareRecHits_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareRecHits_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase1CompareTracks_cfi import *
from DQM.SiPixelHeterogeneous.siPixelPhase2CompareTracks_cfi import *
from DQM.SiPixelHeterogeneous.siPixelHIonPhase1CompareTracks_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareVertices_cfi import *

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

### Alpaka

# digi errors
SiPixelPhase1RawDataConfForSerial = copy.deepcopy(SiPixelPhase1RawDataConf)
for pset in SiPixelPhase1RawDataConfForSerial:
    pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsCPU"

siPixelPhase1MonitorRawDataASerial = SiPixelPhase1RawDataAnalyzer.clone(
    src = "siPixelDigiErrorsAlpakaSerial",
    histograms = SiPixelPhase1RawDataConfForSerial
)

SiPixelPhase1RawDataConfForDevice = copy.deepcopy(SiPixelPhase1RawDataConf)
for pset in SiPixelPhase1RawDataConfForDevice:
    pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsGPU"

siPixelPhase1MonitorRawDataADevice = SiPixelPhase1RawDataAnalyzer.clone(
    src = "siPixelDigiErrorsAlpaka",
    histograms = SiPixelPhase1RawDataConfForDevice
)

siPixelPhase1CompareDigiErrorsSoAAlpaka = siPixelPhase1RawDataErrorComparator.clone(
    pixelErrorSrcGPU = cms.InputTag("siPixelDigiErrorsAlpaka"),
    pixelErrorSrcCPU = cms.InputTag("siPixelDigiErrorsAlpakaSerial"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelErrorCompareGPUvsCPU')
)

# PixelRecHits: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelRecHitsSoAMonitorSerial = siPixelPhase1MonitorRecHitsSoAAlpaka.clone(
    pixelHitsSrc = cms.InputTag( 'siPixelRecHitsPreSplittingAlpakaSerial' ),
    TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsSerial' )
)

# PixelRecHits: monitor of Device product (Alpaka backend: '')
siPixelRecHitsSoAMonitorDevice = siPixelPhase1MonitorRecHitsSoAAlpaka.clone(
    pixelHitsSrc = cms.InputTag( 'siPixelRecHitsPreSplittingAlpaka' ),
    TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsDevice' )
)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorSerial = siPixelPhase1MonitorTrackSoAAlpaka.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpakaSerial'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackSerial')
)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorDevice = siPixelPhase1MonitorTrackSoAAlpaka.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpaka'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackDevice')
)

# PixelVertices: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelVertexSoAMonitorSerial = siPixelMonitorVertexSoAAlpaka.clone(
    pixelVertexSrc = cms.InputTag("pixelVerticesAlpakaSerial"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexSerial')
)

siPixelVertexSoAMonitorDevice = siPixelMonitorVertexSoAAlpaka.clone(
    pixelVertexSrc = cms.InputTag("pixelVerticesAlpaka"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexDevice')
)

monitorPixelTracksAlpaka = cms.Sequence(siPixelTrackSoAMonitorSerial *
                                        siPixelTrackSoAMonitorDevice *
                                        siPixelPhase1CompareTrackSoA)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorSerialStrip = siPixelPhase1StripMonitorTrackSoA.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpakaSerial'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackSerial')
)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorDeviceStrip = siPixelPhase1StripMonitorTrackSoA.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpaka'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackDevice')
)

monitorPixelTracksAlpakaStrip = cms.Sequence( siPixelTrackSoAMonitorSerialStrip *
                                              siPixelTrackSoAMonitorDeviceStrip *
                                              siPixelPhase1StripCompareTrackSoA)
                                              

from Configuration.ProcessModifiers.stripNtupletFit_cff import stripNtupletFit
stripNtupletFit.toReplaceWith(monitorPixelTracksAlpaka, monitorPixelTracksAlpakaStrip)
stripNtupletFit.toReplaceWith(siPixelPhase1CompareTrackSoA, siPixelPhase1StripCompareTrackSoA)
stripNtupletFit.toReplaceWith(siPixelTrackSoAMonitorSerial, siPixelTrackSoAMonitorSerial)
stripNtupletFit.toReplaceWith(siPixelTrackSoAMonitorDevice, siPixelTrackSoAMonitorDevice)

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
# and the Alpaka version
monitorpixelSoACompareSourceAlpaka = cms.Sequence(
                                            siPixelPhase1MonitorRawDataASerial *
                                            siPixelPhase1MonitorRawDataADevice *
                                            siPixelPhase1CompareDigiErrorsSoAAlpaka *
                                            siPixelRecHitsSoAMonitorSerial *
                                            siPixelRecHitsSoAMonitorDevice *
                                            siPixelPhase1CompareRecHits *
                                            siPixelTrackSoAMonitorSerial *
                                            siPixelTrackSoAMonitorDevice *
                                            siPixelPhase1CompareTracks *
                                            siPixelVertexSoAMonitorSerial *
                                            siPixelVertexSoAMonitorDevice *
                                            siPixelCompareVertices )

# Phase-2 sequence ...
_monitorpixelSoACompareSource =  cms.Sequence(siPixelPhase2MonitorRecHitsSoACPU *
                                              siPixelPhase2MonitorRecHitsSoAGPU *
                                              siPixelPhase2CompareRecHitsSoA *
                                              siPixelPhase2MonitorTrackSoAGPU *
                                              siPixelPhase2MonitorTrackSoACPU *
                                              siPixelPhase2CompareTrackSoA *
                                              siPixelMonitorVertexSoACPU *
                                              siPixelMonitorVertexSoAGPU *
                                              siPixelCompareVertexSoA)

# ...and the Alpaka version
_monitorpixelSoACompareSourceAlpakaPhase2 = cms.Sequence(                                          
                                            siPixelRecHitsSoAMonitorSerial *
                                            siPixelRecHitsSoAMonitorDevice *
                                            siPixelPhase1CompareRecHits *
                                            siPixelTrackSoAMonitorSerial *
                                            siPixelTrackSoAMonitorDevice *
                                            siPixelPhase1CompareTracks *
                                            siPixelVertexSoAMonitorSerial *
                                            siPixelVertexSoAMonitorDevice *
                                            siPixelCompareVertices )

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
phase2_tracker.toReplaceWith(monitorpixelSoACompareSourceAlpaka,_monitorpixelSoACompareSourceAlpakaPhase2)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.alpakaValidationPixel_cff import alpakaValidationPixel
(alpakaValidationPixel & ~gpuValidationPixel ).toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSourceAlpaka)


