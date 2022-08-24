import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_RealData_cfi import *

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

from RecoLocalTracker.SiStripClusterizer.siStripClusterizerFromRawGPU_cfi import siStripClusterizerFromRawGPU
from RecoLocalTracker.SiStripClusterizer.siStripClustersSOAtoHost_cfi import siStripClustersSOAtoHost
from RecoLocalTracker.SiStripClusterizer.siStripClustersFromSOA_cfi import siStripClustersFromSOA
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizerConditionsGPUESProducer_cfi import SiStripClusterizerConditionsGPUESProducer

_siStripClusterizerFromRaw = cms.EDProducer("SiStripClusterizerFromRaw",
                                            onDemand = cms.bool(True),
                                            Clusterizer = DefaultClusterizer,
                                            Algorithms = DefaultAlgorithms,
                                            DoAPVEmulatorCheck = cms.bool(False),
                                            HybridZeroSuppressed = cms.bool(False),
                                            ProductLabel = cms.InputTag('rawDataCollector'))

_siStripClusterizerFromRaw.Clusterizer.MaxClusterSize = cms.uint32(16)

siStripClusterizerFromRawGPU.Clusterizer = DefaultClusterizer

siStripClusters = SwitchProducerCUDA(
    cpu = _siStripClusterizerFromRaw.clone(),
)

siStripClustersTask = cms.Task(
    siStripClusters,
)

from Configuration.ProcessModifiers.gpu_cff import gpu

gpu.toModify(siStripClusters,
    cuda = siStripClustersFromSOA,
)

siStripClustersTaskCUDA = cms.Task()

gpu.toReplaceWith(siStripClustersTaskCUDA, cms.Task(
    # conditions used *only* by the modules running on GPU
    SiStripClusterizerConditionsGPUESProducer,
    # reconstruct the strip clusters on the gpu
    siStripClusterizerFromRawGPU,
    # copy clusters from GPU to pinned host memory
    siStripClustersSOAtoHost,
))

gpu.toReplaceWith(siStripClustersTask, cms.Task(
    # CUDA specific
    siStripClustersTaskCUDA,
    # switch producer to legacy format
    siStripClusters,
))
