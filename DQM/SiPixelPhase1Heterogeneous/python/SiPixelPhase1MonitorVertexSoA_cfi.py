import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
#from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

monitorpixelVertexSoA = DQMEDAnalyzer('SiPixelPhase1MonitorVertexSoA',
                                      pixelVertexSrc = cms.InputTag("pixelVerticesSoA@cpu"),
                                      beamSpotSrc = cms.InputTag("offlineBeamSpot"),
                                      TopFolderName = cms.string("SiPixelHeterogeneous/PixelVertexSoA"),
)

gpu.toModify(monitorpixelVertexSoA,
    pixelVertexSrc = cms.InputTag("pixelVerticesSoA@cuda")
)
