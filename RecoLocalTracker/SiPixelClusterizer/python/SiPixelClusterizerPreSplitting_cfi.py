import FWCore.ParameterSet.Config as cms

from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
siPixelClustersPreSplitting = SwitchProducerCUDA(
    cpu = _siPixelClusters.clone()
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(siPixelClustersPreSplitting,
    cuda = cms.EDAlias(
        siPixelDigisClustersPreSplitting = cms.VPSet(
            cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
        )
    )
)
