from HeterogeneousCore.SonicTriton.TritonService_cfi import *

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

enableSonicTriton.toModify(TritonService,
    fallback = dict(
        enable = True,
    ),
)
