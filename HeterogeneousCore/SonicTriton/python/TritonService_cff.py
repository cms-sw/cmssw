from HeterogeneousCore.SonicTriton.TritonService_cfi import *

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

_gpu_available_cached = None

def _gpu_available():
    global _gpu_available_cached
    if _gpu_available_cached is None:
        import os
        _gpu_available_cached = (os.system("nvidia-smi -L") == 0)
    return _gpu_available_cached

enableSonicTriton.toModify(TritonService,
    fallback = dict(
        enable = True,
        useGPU = _gpu_available(),
    ),
)
