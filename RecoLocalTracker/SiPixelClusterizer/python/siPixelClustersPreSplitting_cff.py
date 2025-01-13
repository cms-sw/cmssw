import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# HIon Modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
# Phase 2 Tracker Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# The legacy pixel cluster producer
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting

siPixelClustersPreSplittingTask = cms.Task(
    siPixelClustersPreSplitting
)

######################################################################

### Alpaka Pixel Clusters Reco

#from CalibTracker.SiPixelESProducers.siPixelCablingSoAESProducer_cfi import siPixelCablingSoAESProducer
#from CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTSoAESProducer_cfi import siPixelGainCalibrationForHLTSoAESProducer

def _addProcessCalibTrackerAlpakaES(process):
    process.load("CalibTracker.SiPixelESProducers.siPixelCablingSoAESProducer_cfi")
    process.load("CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTSoAESProducer_cfi")

modifyConfigurationCalibTrackerAlpakaES_ = alpaka.makeProcessModifier(_addProcessCalibTrackerAlpakaES)

# reconstruct the pixel digis and clusters with alpaka on the device
from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterPhase1_cfi import siPixelRawToClusterPhase1 as _siPixelRawToClusterAlpaka
from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterHIonPhase1_cfi import siPixelRawToClusterHIonPhase1 as _siPixelRawToClusterAlpakaHIonPhase1

siPixelClustersPreSplittingAlpaka = _siPixelRawToClusterAlpaka.clone()

(alpaka & pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelClustersPreSplittingAlpaka,_siPixelRawToClusterAlpakaHIonPhase1.clone())

(alpaka & run3_common).toModify(siPixelClustersPreSplittingAlpaka,
    # use the pixel channel calibrations scheme for Run 3
    clusterThreshold_layer1 = 4000,
    VCaltoElectronGain      = 1,  # all gains=1, pedestals=0
    VCaltoElectronGain_L1   = 1,
    VCaltoElectronOffset    = 0,
    VCaltoElectronOffset_L1 = 0)

from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToCluster_cfi import siPixelPhase2DigiToCluster as _siPixelPhase2DigiToCluster

# for phase2 no pixel raw2digi is available at the moment
# so we skip the raw2digi step and run on pixel digis copied to gpu
from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import PixelDigitizerAlgorithmCommon

(alpaka & phase2_tracker).toReplaceWith(siPixelClustersPreSplittingAlpaka, _siPixelPhase2DigiToCluster.clone(
    Phase2ReadoutMode = PixelDigitizerAlgorithmCommon.Phase2ReadoutMode.value(), # flag to decide Readout Mode : linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4 ...) with threshold subtraction
    Phase2DigiBaseline = int(PixelDigitizerAlgorithmCommon.ThresholdInElectrons_Barrel.value()), # same for barrel and endcap
    Phase2KinkADC = 8,
    ElectronPerADCGain = PixelDigitizerAlgorithmCommon.ElectronPerAdc.value()
))

# reconstruct the pixel digis and clusters with alpaka on the cpu, for validation
siPixelClustersPreSplittingAlpakaSerial = makeSerialClone(siPixelClustersPreSplittingAlpaka)

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase1_cfi import siPixelDigisClustersFromSoAAlpakaPhase1 as _siPixelDigisClustersFromSoAAlpakaPhase1
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase2_cfi import siPixelDigisClustersFromSoAAlpakaPhase2 as _siPixelDigisClustersFromSoAAlpakaPhase2
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaHIonPhase1_cfi import siPixelDigisClustersFromSoAAlpakaHIonPhase1 as _siPixelDigisClustersFromSoAAlpakaHIonPhase1

alpaka.toReplaceWith(siPixelClustersPreSplitting,_siPixelDigisClustersFromSoAAlpakaPhase1.clone(
    src = "siPixelClustersPreSplittingAlpaka"
))

(alpaka & pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelClustersPreSplitting,_siPixelDigisClustersFromSoAAlpakaHIonPhase1.clone(
    src = "siPixelClustersPreSplittingAlpaka"
))

(alpaka & phase2_tracker).toReplaceWith(siPixelClustersPreSplitting,_siPixelDigisClustersFromSoAAlpakaPhase2.clone(
    clusterThreshold_layer1 = 4000,
    clusterThreshold_otherLayers = 4000,
    src = "siPixelClustersPreSplittingAlpaka",
    storeDigis = False,
    produceDigis = False
))

# These produce pixelDigiErrors in Alpaka; they are constructed here because they need
# siPixelClustersPreSplittingAlpaka* as input
from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoAAlpaka_cfi import siPixelDigiErrorsFromSoAAlpaka as _siPixelDigiErrorsFromSoAAlpaka
siPixelDigiErrorsAlpaka = _siPixelDigiErrorsFromSoAAlpaka.clone(
    digiErrorSoASrc = cms.InputTag('siPixelClustersPreSplittingAlpaka'),
    fmtErrorsSoASrc = cms.InputTag('siPixelClustersPreSplittingAlpaka'),
    UsePhase1 = cms.bool(True)
)

siPixelDigiErrorsAlpakaSerial = siPixelDigiErrorsAlpaka.clone(
    digiErrorSoASrc = cms.InputTag('siPixelClustersPreSplittingAlpakaSerial'),
    fmtErrorsSoASrc = cms.InputTag('siPixelClustersPreSplittingAlpakaSerial')
)

# Run 3
alpaka.toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
    # reconstruct the pixel clusters with alpaka
    siPixelClustersPreSplittingAlpaka,
    # reconstruct the pixel clusters with alpaka on the cpu (if requested by the validation)
    siPixelClustersPreSplittingAlpakaSerial,
    # reconstruct pixel digis errors legacy with alpaka on serial and device
    siPixelDigiErrorsAlpaka,
    siPixelDigiErrorsAlpakaSerial,
    # convert from host SoA to legacy formats (digis and clusters)
    siPixelClustersPreSplitting
))

# Phase 2
(alpaka & phase2_tracker).toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
    # reconstruct the pixel clusters with alpaka from copied digis
    siPixelClustersPreSplittingAlpaka,
    # reconstruct the pixel clusters with alpaka from copied digis on the cpu (if requested by the validation)
    siPixelClustersPreSplittingAlpakaSerial,
    # reconstruct pixel digis errors legacy with alpaka on serial and device
    siPixelDigiErrorsAlpaka,
    siPixelDigiErrorsAlpakaSerial,
    # convert the pixel digis (except errors) and clusters to the legacy format
    siPixelClustersPreSplitting
))
