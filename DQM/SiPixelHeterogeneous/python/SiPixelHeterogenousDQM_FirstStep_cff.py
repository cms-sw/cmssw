import copy
import FWCore.ParameterSet.Config as cms
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
monitorpixelSoASource = cms.Sequence()
# Run-3 Alpaka sequence 
monitorpixelSoASourceAlpaka = cms.Sequence(siPixelPhase1MonitorRecHitsSoAAlpaka * siPixelPhase1MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
alpaka.toReplaceWith(monitorpixelSoASource, monitorpixelSoASourceAlpaka)
# Phase-2 sequence
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_monitorpixelSoARecHitsSource = cms.Sequence()
(phase2_tracker & ~alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSource)
_monitorpixelSoARecHitsSourceAlpaka = cms.Sequence(siPixelPhase2MonitorRecHitsSoAAlpaka * siPixelPhase2MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
(phase2_tracker & alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceAlpaka)

# HIon Phase 1 sequence
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

_monitorpixelSoARecHitsSourceHIon = cms.Sequence()
(pp_on_AA & ~phase2_tracker).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceHIon)
_monitorpixelSoARecHitsSourceHIonAlpaka = cms.Sequence(siPixelHIonPhase1MonitorRecHitsSoAAlpaka * siPixelHIonPhase1MonitorTrackSoAAlpaka * siPixelMonitorVertexSoAAlpaka)
(pp_on_AA & ~phase2_tracker & alpaka).toReplaceWith(monitorpixelSoASource, _monitorpixelSoARecHitsSourceHIonAlpaka)

#Define the sequence for GPU vs CPU validation
#This should run:- individual monitor for the 2 collections + comparison module
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

# Run-3 sequence
monitorpixelSoACompareSource = cms.Sequence(siPixelPhase1MonitorRawDataACPU *
                                            siPixelPhase1MonitorRawDataAGPU *
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
_monitorpixelSoACompareSource =  cms.Sequence()

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

phase2_tracker.toReplaceWith(monitorpixelSoACompareSource,_monitorpixelSoACompareSource)
phase2_tracker.toReplaceWith(monitorpixelSoACompareSourceAlpaka,_monitorpixelSoACompareSourceAlpakaPhase2)

from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSource)

from Configuration.ProcessModifiers.alpakaValidationPixel_cff import alpakaValidationPixel
(alpakaValidationPixel & ~gpuValidationPixel ).toReplaceWith(monitorpixelSoASource, monitorpixelSoACompareSourceAlpaka)


