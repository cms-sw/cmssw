import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# SiPixelGainCalibrationServiceParameters
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *

# legacy pixel cluster producer
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
siPixelClustersPreSplitting = SwitchProducerCUDA(
    cpu = _siPixelClusters.clone()
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.ProcessModifiers.gpu_cff import gpu

# ensure the same results when running on GPU (which supports only the 'HLT' payload) and CPU
# but not for phase-2 where we don't calibrate digis in the clusterizer (yet)
(gpu & ~phase2_tracker).toModify(siPixelClustersPreSplitting,
    cpu = dict(
        payloadType = 'HLT'
    )
)

# SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
gpu.toModify(siPixelClustersPreSplitting,
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
