import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import pvClusterComparer, pixelVertices as _pixelVertices
pixelVertices = SwitchProducerCUDA(
    cpu = _pixelVertices.clone()
)
