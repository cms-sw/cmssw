import FWCore.ParameterSet.Config as cms
from EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi import siPixelRawToDigi as _siPixelRawToDigi

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
siPixelDigis = SwitchProducerCUDA(
    cpu = _siPixelRawToDigi.clone()
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis.cpu, UsePhase1=True)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(siPixelDigis,
    cuda = cms.EDAlias(
        siPixelDigiErrors = cms.VPSet(
            cms.PSet(type = cms.string("DetIdedmEDCollection")),
            cms.PSet(type = cms.string("SiPixelRawDataErroredmDetSetVector")),
            cms.PSet(type = cms.string("PixelFEDChanneledmNewDetSetVector"))
        ),
        siPixelDigisClustersPreSplitting = cms.VPSet(
            cms.PSet(type = cms.string("PixelDigiedmDetSetVector"))
        )
    )
)
