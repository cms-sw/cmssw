import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# legacy pixel unpacker
from EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi import siPixelRawToDigi as _siPixelRawToDigi
siPixelDigis = SwitchProducerCUDA(
    cpu = _siPixelRawToDigi.clone()
)

# use the Phase 1 settings
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis.cpu,
    UsePhase1 = True
)

from Configuration.ProcessModifiers.siPixelQualityRawToDigi_cff import siPixelQualityRawToDigi
siPixelQualityRawToDigi.toModify(siPixelDigis.cpu,
    UseQualityInfo = True,
    SiPixelQualityLabel = 'forRawToDigi',
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
(gpu & ~phase2_tracker).toModify(siPixelDigis,
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
