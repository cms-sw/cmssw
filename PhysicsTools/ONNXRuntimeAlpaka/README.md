# PhysicsTools/ONNXRuntimeAlpaka

This package runs ONNX Runtime inference directly on SoA data stored in alpaka `PortableCollection`s, in host or
device memory, without copying the data through `std::vector`s as done by
[PhysicsTools/ONNXRuntime](../ONNXRuntime).
It follows the design of [PhysicsTools/PyTorchAlpaka](../PyTorchAlpaka), and provides a similar interface:

```cpp
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h"

// data member of a stream module
ort::AlpakaSession session_{params.getParameter<edm::FileInPath>("model").fullPath()};

// in FixedQueueEDProducer::beginStream(edm::StreamID, Queue queue)
session_.bind(queue);

// in produce()
cms::Ort::alpakatools::TensorCollection<Queue> inputs(total_size);
inputs.add<portabletest::ParticleSoA>("particles", records.pt(), records.eta(), records.phi());
cms::Ort::alpakatools::TensorCollection<Queue> outputs(total_size);
outputs.add<portabletest::SimpleNetSoA>("regression_head", output_records.reco_pt());
session_.forward(event.queue(), inputs, outputs);
```

Examples covering all the features can be found in [PhysicsTools/ONNXRuntimeAlpakaTest](../ONNXRuntimeAlpakaTest).

## Supported backends

| backend | execution | notes |
|---------|-----------|-------|
| `serial_sync` | ONNX Runtime CPU execution provider | single-threaded, in the calling thread |
| `cuda_async` | ONNX Runtime CUDA execution provider | fully asynchronous, in the CUDA stream of the alpaka queue |
| `rocm_async` | ONNX Runtime CPU execution provider | fallback: the tensors are copied to the host and back, with a host synchronisation; compiled, but not tested |

## Streams and queues

PyTorch can switch to a different CUDA stream at each call (see `QueueGuard` in PyTorchAlpaka).
ONNX Runtime cannot: the CUDA stream is set when a session is created, through the `user_compute_stream` option of
the CUDA execution provider, and cannot be changed afterwards (the CUDA execution provider does not support
`SetEpDynamicOptions`, and `RunOptions::SetSyncStream` only accepts streams created by ONNX Runtime itself).
This is still the case in ONNX Runtime 1.30.

For this reason, each `AlpakaSession` is bound to a single alpaka queue:
  - in a `FixedQueueEDProducer`, call `bind(queue)` from `beginStream()`: all the inference calls run directly in the
    queue used by the module;
  - in other modules, the session is bound to the queue of the first event, and the inference is synchronised with
    the queue of each event using alpaka events (`alpaka::wait(queue, event)`), without blocking the host.

`forward()` never blocks the host on the `cuda_async` backend: the session is configured to not synchronise at the
end of `Run()`, and the memory copies are done in the compute stream. This has been verified with Nsight Systems:
inside `InferenceSession::Run` there are only asynchronous CUDA calls, and all the ONNX Runtime kernels run in the
CUDA stream of the alpaka queue.

## Tensors

The `TensorCollection` and `TensorHandle` classes are copies of the PyTorchAlpaka ones, adapted to ONNX Runtime.
The columns are registered in the same way (`SOA_COLUMN`, `SOA_EIGEN_COLUMN`, `SOA_SCALAR`, mini-batches, and
`change_order()`), and the inputs and outputs are matched to the model inputs and outputs by position.

The main difference is that ONNX Runtime only supports contiguous, row-major tensors, while PyTorch supports strided
tensors. Each tensor can be presented to ONNX Runtime in one of two layouts:

  - `Layout::SampleMajor` (default): the tensor has the same shape as in PyTorch, e.g. `[batch, features]` or
    `[batch, 3, 9, 9]`, so the same model works with both frameworks.
    - Tensors made of a single SoA column (e.g. `[batch]` or `[batch, 1]`) are used in place, without any copy.
    - Other tensors are packed into a temporary, contiguous buffer by a small alpaka kernel (and outputs are unpacked
      from it), in the same queue as the inference. This is a device-to-device copy of the same size as the one
      PyTorchAlpaka does for all `const` inputs.
  - `Layout::FeatureMajor`: the SoA memory is used directly as a `[columns, padded size]` tensor, without any copy.
    The model must accept and return transposed tensors (e.g. wrapping it as in `SimpleNet.py` in
    [PhysicsTools/ONNXRuntimeAlpakaTest](../ONNXRuntimeAlpakaTest)), and it will also
    process the padding elements at the end of each column, whose value is undefined; this is suitable only for models
    that process each element independently. This layout requires a tensor spanning the whole collection (no
    mini-batches).
    ```cpp
    inputs.set_layout("particles", cms::Ort::alpakatools::Layout::FeatureMajor);
    outputs.set_layout("regression_head", cms::Ort::alpakatools::Layout::FeatureMajor);
    ```

Compared to PyTorchAlpaka:
  - `const` inputs are never copied: ONNX Runtime does not modify its inputs;
  - models with multiple outputs write all of them directly into the SoA memory, without intermediate copies;
  - the element types of the tensors are checked against the model inputs and outputs.

## Configuration of the CUDA execution provider

See [SessionOptions.h](interface/SessionOptions.h):
  - `use_tf32 = 0`: ONNX Runtime uses TF32 for FP32 matrix multiplications and convolutions by default, which changes
    the results by O(1e-3) with respect to PyTorch and to the CPU;
  - `cudnn_conv_algo_search = HEURISTIC`: avoid benchmarking the convolution algorithms for each new input shape;
  - `arena_extend_strategy = kSameAsRequested`: grow the memory arena only by the amount requested;
  - `do_copy_in_default_stream = 0`: required, because alpaka queues do not synchronise with the legacy default stream.

## Limitations

  - Each session uses its own memory arena for the temporary tensors, outside of the CMSSW caching allocator: ONNX
    Runtime does not allow using an external allocator together with a user compute stream.
  - Each module instance (i.e. each EDM stream) has its own copy of the model weights, like in PyTorchAlpaka.
  - The output shapes must be fully determined by the input shapes, and outputs cannot be scalars.
  - Models with data-dependent output shapes, or operations that need to read device data on the host, will
    synchronise the device internally.
  - Empty collections are skipped, and the outputs are left untouched.
  - CUDA graphs cannot be used, because the addresses of the inputs and outputs change for every event.
  - The ROCm backend falls back to the CPU, because ONNX Runtime in CMSSW is built without the ROCm or MIGraphX
    execution providers.
