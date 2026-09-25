# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""CPU / GPU (alpaka) backend selection.

pyTICL lets a user pick the compute backend per module, *where the module
supports it*.  The validator already rejects a GPU request on a module that has
no alpaka implementation (see :mod:`RecoTICL.Configuration.validator`); this module
implements the mechanism that actually drives a portable (``@alpaka``) module to
CPU (``serial_sync``) or GPU (``cuda_async``) and wires up
``ProcessAcceleratorAlpaka`` on the process.

HGCAL local reconstruction / layer clustering is where this matters today: the
SoA rechit & clustering producers are portable, while the TICL pattern-recognition
producers are CPU-only.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CPU, GPU

# pyTICL backend name -> alpaka backend string
ALPAKA_BACKEND = {CPU: "serial_sync", GPU: "cuda_async"}


def is_alpaka(module):
    """True if ``module`` is a portable ``@alpaka`` producer."""
    return module.type_().endswith("@alpaka")


def set_module_backend(module, backend):
    """Drive a portable ``@alpaka`` ``module`` to ``backend`` ('cpu' or 'gpu')."""
    if not is_alpaka(module):
        from RecoTICL.Configuration.model import PyTICLError
        raise PyTICLError(
            "module type %r is not a portable @alpaka module; it cannot run on a "
            "GPU backend" % module.type_())
    if backend not in ALPAKA_BACKEND:
        from RecoTICL.Configuration.model import PyTICLError
        raise PyTICLError("unknown backend %r (known: %s)"
                          % (backend, ", ".join(ALPAKA_BACKEND)))
    if not hasattr(module, "alpaka"):
        module.alpaka = cms.untracked.PSet()
    module.alpaka.backend = cms.untracked.string(ALPAKA_BACKEND[backend])
    return module


def add_process_accelerator(process, backend=None):
    """Add (idempotently) ``ProcessAcceleratorAlpaka`` to ``process`` and, if a
    ``backend`` is given, set it process-wide."""
    from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka import ProcessAcceleratorAlpaka
    if not hasattr(process, "ProcessAcceleratorAlpaka"):
        process.ProcessAcceleratorAlpaka = ProcessAcceleratorAlpaka()
    if backend is not None:
        process.ProcessAcceleratorAlpaka.setBackend(ALPAKA_BACKEND.get(backend, backend))
    return process.ProcessAcceleratorAlpaka


def apply_backends(assembled, process):
    """After ``assembled.add_to_process(process)``: drive each portable module to
    its iteration's requested backend and ensure ``ProcessAcceleratorAlpaka`` is
    present.  CPU-only modules are left untouched (GPU on them is rejected by the
    validator)."""
    cfg = assembled.config
    requested = {}
    for it in cfg.iterations:
        for lbl in (cfg.target.filter_label(it.name), cfg.target.trackster_label(it.name)):
            requested[lbl] = it.backend
    any_alpaka = False
    for label, module in assembled.modules.items():
        if is_alpaka(module):
            any_alpaka = True
            set_module_backend(module, requested.get(label, CPU))
    if any_alpaka:
        add_process_accelerator(process)
    return process
