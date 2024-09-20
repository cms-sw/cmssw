import FWCore.ParameterSet.Config as cms

# legacy pixel unpacker
from EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi import siPixelRawToDigi as _siPixelRawToDigi
siPixelDigis = _siPixelRawToDigi.clone()

# use the Phase 1 settings
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis,
    UsePhase1 = True
)

from Configuration.ProcessModifiers.siPixelQualityRawToDigi_cff import siPixelQualityRawToDigi
siPixelQualityRawToDigi.toModify(siPixelDigis,
    UseQualityInfo = True,
    SiPixelQualityLabel = 'forRawToDigi',
)
