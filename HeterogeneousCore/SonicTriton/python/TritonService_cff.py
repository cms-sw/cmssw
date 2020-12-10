from HeterogeneousCore.SonicTriton.TritonService_cfi import *

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import _switch_cuda

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

enableSonicTriton.toModify(TritonService,
    fallback = dict(
        enable = True,
        useGPU = _switch_cuda()[0],
    ),
)
