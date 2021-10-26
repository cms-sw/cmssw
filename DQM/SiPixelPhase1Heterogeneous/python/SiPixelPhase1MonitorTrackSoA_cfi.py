import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer


monitorpixelTrackSoA = DQMEDAnalyzer('SiPixelPhase1MonitorTrackSoA',
                                        pixelTrackSrc = cms.InputTag("pixelTracksSoA@cpu"),
                                        TopFolderName = cms.string("SiPixelHeterogeneous/PixelTrackSoA"),
)

gpu.toModify(monitorpixelTrackSoA,
    pixelTrackSrc = cms.InputTag("pixelTracksSoA@cuda")
)

'''
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
monitorpixelTrackSoA =  SwitchProducerCUDA(
    cpu = DQMEDAnalyzer('SiPixelPhase1MonitorTrackSoA',
                            pixelTrackSrc = cms.InputTag("pixelTracksSoA@cpu"),
                            TopFolderName = cms.string("SiPixelHeterogeneous/PixelTrackSoA"),
                        ),
    cuda = DQMEDAnalyzer('SiPixelPhase1MonitorTrackSoA',
                             pixelTrackSrc = cms.InputTag("pixelTracksSoA@cuda"),
                             TopFolderName = cms.string("SiPixelHeterogeneous/PixelTrackSoA"),
                         ),
)
'''
