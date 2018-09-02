import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelClusterizer.siPixelClustersHeterogeneousDefault_cfi import siPixelClustersHeterogeneousDefault as _siPixelClustersHeterogeneousDefault
siPixelClustersHeterogeneous = _siPixelClustersHeterogeneousDefault.clone()

# following copied from SiPixelRawToDigi_cfi
siPixelClustersHeterogeneous.IncludeErrors = cms.bool(True)
siPixelClustersHeterogeneous.InputLabel = cms.InputTag("rawDataCollector")
siPixelClustersHeterogeneous.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
siPixelClustersHeterogeneous.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
siPixelClustersHeterogeneous.UserErrorList = cms.vint32(40)
##  Use pilot blades
siPixelClustersHeterogeneous.UsePilotBlade = cms.bool(False)
##  Use phase1
siPixelClustersHeterogeneous.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
siPixelClustersHeterogeneous.Regions = cms.PSet( )
siPixelClustersHeterogeneous.CablingMapLabel = cms.string("")

# The following is copied from siPixelClusters_cfi, clearly not
# maintainable in the long run
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelClustersHeterogeneous,
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
phase1Pixel.toModify(siPixelClustersHeterogeneous, UsePhase1=True)
