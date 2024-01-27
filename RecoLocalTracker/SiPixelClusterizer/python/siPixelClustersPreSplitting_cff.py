import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# conditions used *only* by the modules running on GPU
from CalibTracker.SiPixelESProducers.siPixelROCsStatusAndMappingWrapperESProducer_cfi import siPixelROCsStatusAndMappingWrapperESProducer
from CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTGPU_cfi import siPixelGainCalibrationForHLTGPU

# SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting

siPixelClustersPreSplittingTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
    siPixelClustersPreSplitting
)

# reconstruct the pixel digis and clusters on the gpu
from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDAPhase1_cfi import siPixelRawToClusterCUDAPhase1 as _siPixelRawToClusterCUDA
from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDAHIonPhase1_cfi import siPixelRawToClusterCUDAHIonPhase1 as _siPixelRawToClusterCUDAHIonPhase1

siPixelClustersPreSplittingCUDA = _siPixelRawToClusterCUDA.clone()

# HIon Modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
# Phase 2 Tracker Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

(pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelClustersPreSplittingCUDA, _siPixelRawToClusterCUDAHIonPhase1.clone())

run3_common.toModify(siPixelClustersPreSplittingCUDA,
                     # use the pixel channel calibrations scheme for Run 3
                     clusterThreshold_layer1 = 4000,
                     VCaltoElectronGain      = 1,  # all gains=1, pedestals=0
                     VCaltoElectronGain_L1   = 1,
                     VCaltoElectronOffset    = 0,
                     VCaltoElectronOffset_L1 = 0)

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase1_cfi import siPixelDigisClustersFromSoAPhase1 as _siPixelDigisClustersFromSoAPhase1
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase2_cfi import siPixelDigisClustersFromSoAPhase2 as _siPixelDigisClustersFromSoAPhase2

siPixelDigisClustersPreSplitting = _siPixelDigisClustersFromSoAPhase1.clone()

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAHIonPhase1_cfi import siPixelDigisClustersFromSoAHIonPhase1 as _siPixelDigisClustersFromSoAHIonPhase1
(pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelDigisClustersPreSplitting, _siPixelDigisClustersFromSoAHIonPhase1.clone())


run3_common.toModify(siPixelDigisClustersPreSplitting,
                     clusterThreshold_layer1 = 4000)

gpu.toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
    # conditions used *only* by the modules running on GPU
    siPixelROCsStatusAndMappingWrapperESProducer,
    siPixelGainCalibrationForHLTGPU,
    # reconstruct the pixel digis and clusters on the gpu
    siPixelClustersPreSplittingCUDA,
    # convert the pixel digis (except errors) and clusters to the legacy format
    siPixelDigisClustersPreSplitting,
    # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
    siPixelClustersPreSplittingTask.copy()
))

from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToClusterCUDA_cfi import siPixelPhase2DigiToClusterCUDA as _siPixelPhase2DigiToClusterCUDA
# for phase2 no pixel raw2digi is available at the moment
# so we skip the raw2digi step and run on pixel digis copied to gpu

from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import PixelDigitizerAlgorithmCommon

phase2_tracker.toReplaceWith(siPixelClustersPreSplittingCUDA,_siPixelPhase2DigiToClusterCUDA.clone(
  Phase2ReadoutMode = PixelDigitizerAlgorithmCommon.Phase2ReadoutMode.value(), # Flag to decide Readout Mode : linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4 ...) with threshold subtraction
  Phase2DigiBaseline = int(PixelDigitizerAlgorithmCommon.ThresholdInElectrons_Barrel.value()), #Same for barrel and endcap
  Phase2KinkADC = 8,
  ElectronPerADCGain = PixelDigitizerAlgorithmCommon.ElectronPerAdc.value()
))

from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
siPixelDigisPhase2SoA = _siPixelDigisSoAFromCUDA.clone(
    src = "siPixelClustersPreSplittingCUDA"
)

phase2_tracker.toReplaceWith(siPixelDigisClustersPreSplitting, _siPixelDigisClustersFromSoAPhase2.clone(
                        clusterThreshold_layer1 = 4000,
                        clusterThreshold_otherLayers = 4000,
                        src = "siPixelDigisPhase2SoA",
                        #produceDigis = False
                        ))

(gpu & phase2_tracker).toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
                            # reconstruct the pixel clusters on the gpu from copied digis
                            siPixelClustersPreSplittingCUDA,
                            # copy from gpu to cpu
                            siPixelDigisPhase2SoA,
                            # convert the pixel digis (except errors) and clusters to the legacy format
                            siPixelDigisClustersPreSplitting,
                            # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
                            siPixelClustersPreSplitting))

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
siPixelClustersPreSplittingAlpaka = _siPixelRawToClusterAlpaka.clone()

(alpaka & run3_common).toModify(siPixelClustersPreSplittingAlpaka,
    # use the pixel channel calibrations scheme for Run 3
    clusterThreshold_layer1 = 4000,
    VCaltoElectronGain      = 1,  # all gains=1, pedestals=0
    VCaltoElectronGain_L1   = 1,
    VCaltoElectronOffset    = 0,
    VCaltoElectronOffset_L1 = 0)

from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToCluster_cfi import siPixelPhase2DigiToCluster as _siPixelPhase2DigiToCluster

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

(alpaka & ~phase2_tracker).toReplaceWith(siPixelDigisClustersPreSplitting,_siPixelDigisClustersFromSoAAlpakaPhase1.clone(
    src = "siPixelClustersPreSplittingAlpaka"
))

(alpaka & phase2_tracker).toReplaceWith(siPixelDigisClustersPreSplitting,_siPixelDigisClustersFromSoAAlpakaPhase2.clone(
    clusterThreshold_layer1 = 4000,
    clusterThreshold_otherLayers = 4000,
    src = "siPixelClustersPreSplittingAlpaka",
    storeDigis = False,
    produceDigis = False
))

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase1_cfi import siPixelDigisClustersFromSoAAlpakaPhase1 as _siPixelDigisClustersFromSoAAlpakaPhase1
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase2_cfi import siPixelDigisClustersFromSoAAlpakaPhase2 as _siPixelDigisClustersFromSoAAlpakaPhase2

alpaka.toModify(siPixelClustersPreSplitting,
    cpu = cms.EDAlias(
        siPixelDigisClustersPreSplitting = cms.VPSet(
            cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
        )
    )
)

# Run 3
alpaka.toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
    # reconstruct the pixel clusters with alpaka
    siPixelClustersPreSplittingAlpaka,
    # reconstruct the pixel clusters with alpaka on the cpu (if requested by the validation)
    siPixelClustersPreSplittingAlpakaSerial,
    # convert from host SoA to legacy formats (digis and clusters)
    siPixelDigisClustersPreSplitting,
    # EDAlias for the clusters
    siPixelClustersPreSplitting)
)

# Phase 2
(alpaka & phase2_tracker).toReplaceWith(siPixelClustersPreSplittingTask, cms.Task(
    # reconstruct the pixel clusters with alpaka from copied digis
    siPixelClustersPreSplittingAlpaka,
    # reconstruct the pixel clusters with alpaka from copied digis on the cpu (if requested by the validation)
    siPixelClustersPreSplittingAlpakaSerial,
    # convert the pixel digis (except errors) and clusters to the legacy format
    siPixelDigisClustersPreSplitting,
    # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
    siPixelClustersPreSplitting)
)
