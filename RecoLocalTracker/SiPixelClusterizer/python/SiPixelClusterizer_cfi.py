import FWCore.ParameterSet.Config as cms

#
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
siPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    SiPixelGainCalibrationServiceParameters,
    src = cms.InputTag("siPixelDigis"),
    ChannelThreshold = cms.int32(2500),
    MissCalibrate = cms.untracked.bool(True),
    VCaltoElectronGain = cms.int32(65),
    VCaltoElectronOffset = cms.int32(0),                          
    # **************************************
    # ****  payLoadType Options         ****
    # ****  HLT - column granularity    ****
    # ****  Offline - gain:col/ped:pix  ****
    # **************************************
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(3000),
    ClusterThreshold = cms.double(5050.0)
)


