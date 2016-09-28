
import FWCore.ParameterSet.Config as cms

#
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
phase2ITPixelClusters = cms.EDProducer("Phase2ITPixelClusterProducer",
    SiPixelGainCalibrationServiceParameters,
    src = cms.InputTag("siPixelDigis"),
    ChannelThreshold = cms.int32(1000),
    MissCalibrate = cms.untracked.bool(True),
    SplitClusters = cms.bool(False),
    VCaltoElectronGain = cms.int32(65),
    VCaltoElectronOffset = cms.int32(-414),                          
    # **************************************
    # ****  payLoadType Options         ****
    # ****  HLT - column granularity    ****
    # ****  Offline - gain:col/ped:pix  ****
    # **************************************
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(1000),
    ClusterThreshold = cms.double(4000.0),
    # **************************************
    maxNumberOfClusters = cms.int32(-1), # -1 means no limit.
)

# This customization will be removed once we have phase2 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(phase2ITPixelClusters, #FIXME
    src = cms.InputTag('simSiPixelDigis', "Pixel"),
    MissCalibrate = False
)
