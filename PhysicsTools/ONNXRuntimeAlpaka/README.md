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
cms::Ort::alpakatools::TensorCollection<Queue> inputs;
inputs.add<portabletest::ParticleSoA>("particles", records.pt(), records.eta(), records.phi());
cms::Ort::alpakatools::TensorCollection<Queue> outputs;
outputs.add<portabletest::SimpleNetSoA>("regression_head", output_records.reco_pt());
session_.forward(event.queue(), inputs, outputs);
```

Examples covering all the features can be found in [PhysicsTools/ONNXRuntimeAlpakaTest](../ONNXRuntimeAlpakaTest).

## Supported backends

| backend | execution | notes |
|---------|-----------|-------|
| `serial_sync` | ONNX Runtime CPU execution provider | single-threaded, in the calling thread |
| `cuda_async` | ONNX Runtime CUDA execution provider | fully asynchronous, in the CUDA stream of the alpaka queue |
| `rocm_async` | ONNX Runtime MIGraphX execution provider | asynchronous, in the HIP stream of the alpaka queue; requires the `user_compute_stream` option of the MIGraphX execution provider, added to the CMSSW build of ONNX Runtime |

## Streams and queues

PyTorch can switch to a different CUDA stream at each call (see `QueueGuard` in PyTorchAlpaka).
ONNX Runtime cannot: the CUDA or HIP stream is set when a session is created, through the `user_compute_stream` option
of the CUDA or MIGraphX execution provider, and cannot be changed afterwards (the CUDA execution provider does not support
`SetEpDynamicOptions`, and `RunOptions::SetSyncStream` only accepts streams created by ONNX Runtime itself).
This is still the case in ONNX Runtime 1.30.

For this reason, each `AlpakaSession` is bound to a single alpaka queue:
  - in a `FixedQueueEDProducer`, call `bind(queue)` from `beginStream()`: all the inference calls run directly in the
    queue used by the module;
  - in other modules, if `bind()` has not been called, the first call to `forward()` binds the session to the queue of
    the event. When `forward()` is later called with a different queue, the inference runs in the bound queue, and is
    synchronised with the queue of the event using alpaka events (`alpaka::wait(queue, event)`), without blocking the
    host.

The synchronisation is done in each call to `forward()` with a queue different from the bound one, so a loop over
mini-batches records and waits for two alpaka events per batch. These operations are asynchronous and cheap, but can
be avoided altogether by using a `FixedQueueEDProducer`.

`forward()` never blocks the host on the `cuda_async` backend: the session is configured to not synchronise at the
end of `Run()`, and the memory copies are done in the compute stream. This has been verified with Nsight Systems:
inside `InferenceSession::Run` there are only asynchronous CUDA calls, and all the ONNX Runtime kernels run in the
CUDA stream of the alpaka queue.
The same holds for the `rocm_async` backend, with the caveats listed below.
The exception are models with convolutions, when the shape of the inputs changes: see
[Inputs of varying shape](#inputs-of-varying-shape).

## Tensors

The `TensorCollection` and `TensorHandle` classes are copies of the PyTorchAlpaka ones, adapted to ONNX Runtime.
The columns are registered in the same way (`SOA_COLUMN`, `SOA_EIGEN_COLUMN`, `SOA_SCALAR`, `TensorSlice`, and
`change_order()`).

The tensors are matched to the inputs and outputs of the model by name, if the names given to `add()` are all names
of inputs (outputs) of the model, independently of the order in which they are added. If none of the names is used
by the model, the tensors are matched by position, in the order in which they are added or set by `change_order()`,
as in PyTorchAlpaka. If only some of the names match, an exception is thrown, because this usually means that a name
is wrong. The names of the model inputs and outputs are given by `inputNames()` and `outputNames()`, after the session
has been bound, or by any ONNX model viewer.
The size of each tensor is taken from its own SoA, so tensors from SoAs with a different number of elements can be
used together.

### Mini-batches

To run the inference in mini-batches, create a `TensorCollection` for each batch, and pass a
`TensorSlice{batch_id, batch_size}` to select the rows of each tensor; the last batch can contain fewer rows:
```cpp
const uint32_t n_batches = (total_size + batch_size - 1) / batch_size;
for (uint32_t batch_id = 0; batch_id < n_batches; ++batch_id) {
  cms::Ort::alpakatools::TensorCollection<Queue> inputs;
  inputs.add<portabletest::ParticleSoA>("particles", TensorSlice{batch_id, batch_size}, records.pt(), records.eta(), records.phi());
  cms::Ort::alpakatools::TensorCollection<Queue> outputs;
  outputs.add<portabletest::SimpleNetSoA>("regression_head", TensorSlice{batch_id, batch_size}, output_records.reco_pt());
  session_.forward(event.queue(), inputs, outputs);
}
```
The `BatchedTensorCollection` of PyTorchAlpaka is not supported yet.

### Scalars

An `SOA_SCALAR` is presented to ONNX Runtime as a one-element tensor of shape `[1]`, used in place, independently of
the size of the SoA and of the mini-batches. Elementwise operations broadcast it to all rows, e.g. `x * scale` with
`x` of shape `[batch, 3]`; models that need it as a per-row feature (e.g. to concatenate it with other features)
should expand it explicitly. PyTorchAlpaka presents a scalar as a `[size, 1]` tensor with stride 0 instead.

### Layouts

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
    The padded size depends on the size of the elements, because the SoA pads each column to its alignment (128 bytes
    by default): a column of 35 elements is padded to 64 `float`s, to 48 `double`s or `int64_t`s, and to 128
    `uint8_t`s. Tensors with elements of different sizes, or from SoAs of different sizes, have different padded
    sizes, so a model whose outputs have the shape of its inputs can use the FeatureMajor layout only for inputs and
    outputs with elements of the same size, from SoAs with the same number of elements.
    The features of an `SOA_EIGEN_COLUMN` follow the Eigen storage order: for an `R`x`C` matrix, column-major by
    default, element `(r, c)` of the `k`-th matrix is the feature `(k * C + c) * R + r`. This is different from the
    SampleMajor layout, where the same tensor has the shape `[batch, k, R, C]`.
    ```cpp
    inputs.set_layout("particles", cms::Ort::alpakatools::Layout::FeatureMajor);
    outputs.set_layout("regression_head", cms::Ort::alpakatools::Layout::FeatureMajor);
    ```

Compared to PyTorchAlpaka:
  - `const` inputs are never copied: ONNX Runtime does not modify its inputs;
  - models with multiple outputs write all of them directly into the SoA memory, without intermediate copies;
  - the element types of the tensors are checked against the model inputs and outputs.

## Configuration of the CUDA execution provider

The session options are set by the `ONNXService` (see
[ONNXService.cc](../ONNXRuntime/plugins/ONNXService.cc)), and by [SessionOptions.h](interface/SessionOptions.h)
for the device and the stream:
  - `use_tf32 = 0`: ONNX Runtime uses TF32 for FP32 matrix multiplications and convolutions by default, which changes
    the results by O(1e-3) with respect to PyTorch and to the CPU (up to 7e-3 for the regression head of
    MultiHeadNet on an NVIDIA L40S, 2e-5 without TF32). TF32 is disabled by the `ONNXService` since
    [cms-sw/cmssw#52487](https://github.com/cms-sw/cmssw/pull/52487), and can be allowed for all sessions with
    `process.ONNXService.allowTF32 = cms.bool(True)`;
  - `arena.extend_strategy = 1` (`kSameAsRequested`): grow the memory arena only by the amount requested;
  - `do_copy_in_default_stream = 0`: required, because alpaka queues do not synchronise with the legacy default stream.

## Inputs of varying shape

ONNX Runtime prepares the execution of each node for the shapes of its inputs, and prepares it again every time they
change. For most operations this is cheap, but for convolutions the CUDA execution provider selects the cuDNN
algorithm again, synchronously, in the calling thread: the default `cudnn_conv_algo_search = EXHAUSTIVE` benchmarks
all the algorithms. This happens every time the shape changes, also for shapes that have been seen before, so with a
different number of elements in each event `forward()` blocks the host for the whole selection.
Setting `cudnn_conv_algo_search = HEURISTIC` does not change this, and `DEFAULT` reduces it only in part.

Time per event of the TinyResNet test model (3 small convolutional blocks) on an NVIDIA L40S, with one EDM stream:

| input batch size                        | all the images | mini-batches of 32 |
|-----------------------------------------|----------------|--------------------|
| 256 in every event                      | 1.6 ms         | 2.4 ms             |
| different in every event (200 to 499)   | 12.9 ms        | 23.1 ms            |
| same, `cudnn_conv_algo_search = DEFAULT`| 9.6 ms         | 12.2 ms            |

With mini-batches, the shape changes twice per event, between the full batches and the last, smaller one.
The models without convolutions are barely affected: SimpleNet takes 0.37 ms per event with a fixed size, and
0.42 ms with a different size in every event.

For models with convolutions, the inputs should have the same shape in every call to `forward()`, e.g. using
fixed-size mini-batches and padding the last one. The same holds for the ROCm backend, where the MIGraphX execution
provider recompiles the whole model for each new shape.

## Limitations

  - Each session uses its own memory arena for the temporary tensors, outside of the CMSSW caching allocator: ONNX
    Runtime does not allow using an external allocator together with a user compute stream.
  - Each module instance (i.e. each EDM stream) has its own copy of the model weights, like in PyTorchAlpaka.
  - The output shapes must be fully determined by the input shapes, and outputs cannot be scalars.
  - Tensors of different sizes are supported, but the `BatchedTensorCollection` of PyTorchAlpaka is not: the
    mini-batches need an explicit loop, with a `TensorSlice` for each batch.
  - Models with data-dependent output shapes, or operations that need to read device data on the host, will
    synchronise the device internally.
  - If any input tensor is empty the inference is skipped, and the outputs are filled with zeros.
  - Models with convolutions are slow on the CUDA backend if the shape of the inputs changes from call to call, see
    [Inputs of varying shape](#inputs-of-varying-shape).
  - CUDA graphs cannot be used, because the addresses of the inputs and outputs change for every event.
  - On the ROCm backend, the MIGraphX execution provider compiles the model for the shapes of the inputs, and
    recompiles it, synchronously, every time they change: the inputs should have the same shape for every event,
    e.g. using fixed-size mini-batches, padding the last one.
  - On the ROCm backend, the MIGraphX execution provider serialises the inference calls of each session with a mutex,
    and keeps its own copy of the model weights, that cannot be shared between sessions.
