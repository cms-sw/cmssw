import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.siPixelDigisHeterogeneousDefault_cfi import siPixelDigisHeterogeneousDefault as _siPixelDigisHeterogeneousDefault
siPixelDigisHeterogeneous = _siPixelDigisHeterogeneousDefault.clone()

# following copied from SiPixelRawToDigi_cfi
siPixelDigisHeterogeneous.IncludeErrors = cms.bool(True)
siPixelDigisHeterogeneous.InputLabel = cms.InputTag("rawDataCollector")
siPixelDigisHeterogeneous.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
siPixelDigisHeterogeneous.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
siPixelDigisHeterogeneous.UserErrorList = cms.vint32(40)
##  Use pilot blades
siPixelDigisHeterogeneous.UsePilotBlade = cms.bool(False)
##  Use phase1
siPixelDigisHeterogeneous.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
siPixelDigisHeterogeneous.Regions = cms.PSet( )
siPixelDigisHeterogeneous.CablingMapLabel = cms.string("")

# The following is copied from siPixelClusters_cfi, clearly not
# maintainable in the long run
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigisHeterogeneous,
  VCaltoElectronGain      = cms.int32(47),   # L2-4: 47 +- 4.7
  VCaltoElectronGain_L1   = cms.int32(50),   # L1:   49.6 +- 2.6
  VCaltoElectronOffset    = cms.int32(-60),  # L2-4: -60 +- 130
  VCaltoElectronOffset_L1 = cms.int32(-670), # L1:   -670 +- 220
  ChannelThreshold        = cms.int32(10),
  SeedThreshold           = cms.int32(1000),
  ClusterThreshold        = cms.int32(4000),
  ClusterThreshold_L1     = cms.int32(2000)
)

# The following is copied from SiPixelRawToDigi_cfi
phase1Pixel.toModify(siPixelDigisHeterogeneous, UsePhase1=True)
