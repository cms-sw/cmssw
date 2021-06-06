import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.ProcessModifiers.gpu_cff import gpu

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
from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDA_cfi import siPixelRawToClusterCUDA as _siPixelRawToClusterCUDA
siPixelClustersPreSplittingCUDA = _siPixelRawToClusterCUDA.clone()

run3_common.toModify(siPixelClustersPreSplittingCUDA,
    # use the pixel channel calibrations scheme for Run 3
    isRun2 = False
)

# convert the pixel digis (except errors) and clusters to the legacy format
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi import siPixelDigisClustersFromSoA as _siPixelDigisClustersFromSoA
siPixelDigisClustersPreSplitting = _siPixelDigisClustersFromSoA.clone()

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
