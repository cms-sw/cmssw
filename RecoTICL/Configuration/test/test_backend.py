#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""M4: CPU/GPU (alpaka) backend selection.

Drives a real portable HGCAL layer-clustering producer (``@alpaka``) to CPU and
to GPU, wires up ProcessAcceleratorAlpaka, and confirms that requesting GPU on a
CPU-only TICL module is rejected.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import backend
from RecoTICL.Configuration.model import PyTICLError


def main():
    ok = True

    # a real portable HGCAL SoA layer-clustering producer ("...@alpaka")
    from HLTrigger.Configuration.HLT_75e33.modules.hltHgcalSoARecHitsProducer_cfi \
        import hltHgcalSoARecHitsProducer

    gpu_mod = backend.set_module_backend(hltHgcalSoARecHitsProducer.clone(), "gpu")
    cpu_mod = backend.set_module_backend(hltHgcalSoARecHitsProducer.clone(), "cpu")
    if gpu_mod.alpaka.backend.value() != "cuda_async":
        print("FAIL: GPU backend not set (got %r)" % gpu_mod.alpaka.backend.value()); ok = False
    elif cpu_mod.alpaka.backend.value() != "serial_sync":
        print("FAIL: CPU backend not set (got %r)" % cpu_mod.alpaka.backend.value()); ok = False
    else:
        print("OK   : portable producer driven to GPU (cuda_async) and CPU (serial_sync)")

    # ProcessAcceleratorAlpaka is wired up on the process
    p = cms.Process("TEST")
    p.MessageLogger = cms.Service("MessageLogger")
    backend.add_process_accelerator(p, "gpu")
    if not hasattr(p, "ProcessAcceleratorAlpaka"):
        print("FAIL: ProcessAcceleratorAlpaka not added"); ok = False
    else:
        print("OK   : ProcessAcceleratorAlpaka added to the process")

    # GPU on a CPU-only module (a TICL trackster producer) is rejected
    from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer
    try:
        backend.set_module_backend(trackstersProducer.clone(), "gpu")
        print("FAIL: GPU on CPU-only module was not rejected"); ok = False
    except PyTICLError:
        print("OK   : GPU on a CPU-only (non-@alpaka) module is rejected")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
