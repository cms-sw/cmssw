import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# SiPixelGainCalibrationServiceParameters
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *

# legacy pixel cluster producer
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
siPixelClustersPreSplitting = SwitchProducerCUDA(
    cpu = _siPixelClusters.clone()
)

from Configuration.ProcessModifiers.gpu_cff import gpu
# SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
gpu.toModify(siPixelClustersPreSplitting,
    # ensure the same results when running on GPU (which supports only the 'HLT' payload) and CPU
    cpu = dict(
        payloadType = 'HLT'
    ),
    cuda = cms.EDAlias(
        siPixelDigisClustersPreSplitting = cms.VPSet(
            cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
        )
    )
)

from Configuration.ProcessModifiers.siPixelDigiMorphing_cff import siPixelDigiMorphing
siPixelDigiMorphing.toModify(
    siPixelClustersPreSplitting,
    cpu = dict(
         src = 'siPixelDigisMorphed'
    )
)
