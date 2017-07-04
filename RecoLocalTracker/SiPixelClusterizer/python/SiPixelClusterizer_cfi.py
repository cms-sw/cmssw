
import FWCore.ParameterSet.Config as cms

#
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
siPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    SiPixelGainCalibrationServiceParameters,
    src = cms.InputTag("siPixelDigis"),
    ChannelThreshold = cms.int32(1000),
    MissCalibrate = cms.untracked.bool(True),
    SplitClusters = cms.bool(False),
    VCaltoElectronGain    = cms.int32(65),
    VCaltoElectronGain_L1 = cms.int32(65),
    VCaltoElectronOffset    = cms.int32(-414),  
    VCaltoElectronOffset_L1 = cms.int32(-414),  
    # **************************************
    # ****  payLoadType Options         ****
    # ****  HLT - column granularity    ****
    # ****  Offline - gain:col/ped:pix  ****
    # **************************************
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(1000),
    ClusterThreshold    = cms.int32(4000),
    ClusterThreshold_L1 = cms.int32(4000),
    # **************************************
    maxNumberOfClusters = cms.int32(-1), # -1 means no limit.
)

# phase1 pixel
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelClusters,
  VCaltoElectronGain      = cms.int32(47),   # L2-4: 47 +- 4.7
  VCaltoElectronGain_L1   = cms.int32(50),   # L1:   49.6 +- 2.6
  VCaltoElectronOffset    = cms.int32(-60),  # L2-4: -60 +- 130
  VCaltoElectronOffset_L1 = cms.int32(-670), # L1:   -670 +- 220
  ChannelThreshold        = cms.int32(250),
  ClusterThreshold_L1     = cms.int32(2000)
)

# Need these until phase2 pixel templates are used
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siPixelClusters, # FIXME
  src = cms.InputTag('simSiPixelDigis', "Pixel"),
  MissCalibrate = False,
  ElectronPerADCGain = cms.double(600.) # it can be changed to something else (e.g. 135e) if needed
)
