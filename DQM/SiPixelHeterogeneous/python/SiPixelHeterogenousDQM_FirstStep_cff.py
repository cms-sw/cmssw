import copy
import FWCore.ParameterSet.Config as cms
# Alpaka Modules
from Configuration.ProcessModifiers.alpaka_cff import alpaka
from DQM.SiPixelHeterogeneous.siPixelMonitorRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorTrackSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelMonitorVertexSoA_cfi import *

# Run-3 sequence
monitorpixelSoASource = cms.Sequence()
# Run-3 Alpaka sequence 
monitorpixelSoASourceAlpaka = cms.Sequence(siPixelMonitorRecHitsSoA * siPixelMonitorTrackSoA * siPixelMonitorVertexSoA)
alpaka.toReplaceWith(monitorpixelSoASource, monitorpixelSoASourceAlpaka)

# Define the sequence for GPU vs CPU validation
# This should run:- individual monitor for the 2 collections + comparison module
from DQM.SiPixelHeterogeneous.siPixelPhase1RawDataErrorComparator_cfi import *
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *

# comparison modules
from DQM.SiPixelHeterogeneous.siPixelCompareRecHitsSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareTracksSoA_cfi import *
from DQM.SiPixelHeterogeneous.siPixelCompareVerticesSoA_cfi import *

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

siPixelPhase1CompareDigiErrorsSoA = siPixelPhase1RawDataErrorComparator.clone(
    pixelErrorSrcGPU = cms.InputTag("siPixelDigiErrorsAlpaka"),
    pixelErrorSrcCPU = cms.InputTag("siPixelDigiErrorsAlpakaSerial"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelErrorCompareGPUvsCPU')
)

# PixelRecHits: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelRecHitsSoAMonitorSerial = siPixelMonitorRecHitsSoA.clone(
    pixelHitsSrc = cms.InputTag( 'siPixelRecHitsPreSplittingAlpakaSerial' ),
    TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsSerial' )
)

# PixelRecHits: monitor of Device product (Alpaka backend: '')
siPixelRecHitsSoAMonitorDevice = siPixelMonitorRecHitsSoA.clone(
    pixelHitsSrc = cms.InputTag( 'siPixelRecHitsPreSplittingAlpaka' ),
    TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsDevice' )
)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorSerial = siPixelMonitorTrackSoA.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpakaSerial'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackSerial')
)

# PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelTrackSoAMonitorDevice = siPixelMonitorTrackSoA.clone(
    pixelTrackSrc = cms.InputTag('pixelTracksAlpaka'),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackDevice')
)

# PixelVertices: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
siPixelVertexSoAMonitorSerial = siPixelMonitorVertexSoA.clone(
    pixelVertexSrc = cms.InputTag("pixelVerticesAlpakaSerial"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexSerial')
)

siPixelVertexSoAMonitorDevice = siPixelMonitorVertexSoA.clone(
    pixelVertexSrc = cms.InputTag("pixelVerticesAlpaka"),
    topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexDevice')
)

# Run-3 sequence ...
monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRawDataACPU *
                                            siPixelPhase1MonitorRawDataAGPU *
                                            siPixelPhase1RawDataErrorComparator)
# ... and the Alpaka version
monitorpixelSoACompareSourceAlpaka = cms.Sequence(siPixelPhase1MonitorRawDataASerial *
                                                  siPixelPhase1MonitorRawDataADevice *
                                                  siPixelPhase1CompareDigiErrorsSoA *
                                                  siPixelRecHitsSoAMonitorSerial *
                                                  siPixelRecHitsSoAMonitorDevice *
                                                  siPixelCompareRecHitsSoA *
                                                  siPixelTrackSoAMonitorSerial *
                                                  siPixelTrackSoAMonitorDevice *
                                                  siPixelCompareTracksSoA *
                                                  siPixelVertexSoAMonitorSerial *
                                                  siPixelVertexSoAMonitorDevice *
                                                  siPixelCompareVerticesSoA )

# Phase-2 sequence ...
_monitorpixelSoACompareSource =  cms.Sequence()

# ... and the Alpaka version
_monitorpixelSoACompareSourceAlpakaPhase2 = cms.Sequence(siPixelRecHitsSoAMonitorSerial *
                                                         siPixelRecHitsSoAMonitorDevice *
                                                         siPixelCompareRecHitsSoA *
                                                         siPixelTrackSoAMonitorSerial *
                                                         siPixelTrackSoAMonitorDevice *
                                                         siPixelCompareTracksSoA *
                                                         siPixelVertexSoAMonitorSerial *
                                                         siPixelVertexSoAMonitorDevice *
                                                         siPixelCompareVerticesSoA )

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,_monitorpixelSoACompareSource)
phase2_tracker.toReplaceWith(monitorpixelSoACompareSourceAlpaka,_monitorpixelSoACompareSourceAlpakaPhase2)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.alpakaValidationPixel_cff import alpakaValidationPixel
(alpakaValidationPixel & ~gpuValidationPixel ).toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSourceAlpaka)
