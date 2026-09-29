# PhysicsTools/PyTorchAlpaka
This package extends the PyTorch implementation and enables seamless integration with the Alpaka-based heterogeneous computing backend, supporting inference workflows with usage of `pytorch` library with `PortableCollection`s objects allowing users to run direct inference with reduced memory footprint. It provides:
- Compatibility with Alpaka device/queue abstractions.
- Single-threading and CUDA stream management are handled by `QueueGuard` objects specialized for each supported backend.

## Interface for Alpaka Modules
All Pytorch based modules should add `PyTorchService` to disable internal torchlib threading. It enforces single-threaded execution on CPU backends.

Examples demonstrating the interoperability of PyTorch with Alpaka in the CMSSW environment can be found in the [PyTorchAlpakaTest](../PyTorchAlpakaTest) directory. The basic test pipeline includes:
- *SimpleNet* composed with few Dense layers, that operate on SoA style portable data structures. It provides also an example for Runtime FP16 conversion.
- *SimpleNetMiniBatch*, providing and example of inference perfomed in mini-batches
- *MaskedNet* shows how to use multiple input data with `Eigen::Vector` and `SOA_SCALAR`
- *TinyResNet* emulate more complex scenario with `Eigen::Matrix` and how one can implement image-like Tensor implementation
- *TinyResNetMiniBatch* to test the inference in mini-batches in a more complex scenario 
- *MultiHeadNet* handle networks that return more than one output tensor
- *TrackHitDeepSet* shows how to register tensors from SoAs with different numbers of elements

## Model behavior

The `Model` wrapper automatically sets the loaded TorchScript module to evaluation mode (`eval()`).

By default, the model is automatically frozen using the `torch::jit::freeze()` function at construction time, either when a device is specified or when the model is first moved.
You can skip this optimization step by setting `auto_freeze=false` when calling the model constructor.
**Important:** Once a model is frozen, it cannot be moved to another device. Attempting to do so will trigger a runtime assertion.

## Direct Inference on SoA 
The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data (or minimal copy overhead).

**Due to the lack of const correctness ensured by PyTorch, `const` data is currently being copied.**

### TensorCollection
The structural information of the inputs/outputs SoA are stored in a `TensorCollection`, a high level object to register column lists from which tensors are created.

Default-construct a `TensorCollection` and add data blocks with `add` to its internal metadata. Each registered tensor is transformed into a PyTorch tensor (without taking ownership) whose size and type are derived from the columns provided. Each `add` uses the size of its own SoA, so one collection can hold tensors with different numbers of elements.

For two example SoAs Templates, which are stored in PortableCollections, columns can be added to `TensorCollection`, by using the Metarecords implementation of SoAs.

- **Input SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoATemplate,
    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
    SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_SCALAR(float, type),
    SOA_SCALAR(int, someNumber));
```
- **Output SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster));
```
- **Get Metarecords from Portable Collections:**
```cpp
PortableCollection<SoA, Device> deviceCollection(total_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(total_size, queue);
fill(queue, deviceCollection);
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();
```
- **For each function call** of `add` (i.e. one tensor), **add the columns** that should be merged into a single tensor. The datatypes must be the same, and the columns must be contiguous. This means, only columns that are defined directly after each other in the SoA layout can be used for the same tensor. However, not all columns of an SoA have to be used. Only those mentioned in the `registry_tensor` are selected for the tensor creation. Any holes in contiguity created by the alignment are automatically taken care of by the stride calculation.

**IMPORTANT:** continuity of memory is a strict requirement!
```cpp
TensorCollection<Queue> input;
input.add<SoA>("eigen_vector", records.a(), records.b());
input.add<SoA>("eigen_matrix", records.c());
input.add<SoA>("column", records.x(), records.y(), records.z());
input.add<SoA>("scalar", records.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

TensorCollection<Queue> output;
output.add<SoA_Result>("result", result_records.cluster());
```

<!-- For Eigen columns, if only a single Vector/Matrix is provided for the tensor, is provided, as if each vector dimension is a column. This means size of tensor is (nElements, dimension) instead of (nElements, 1, dimension). -->
For Eigen column types, providing a single Eigen::Vector or Eigen::Matrix is interpreted as a 2D tensor where each vector entry corresponds to a column. In this case, the resulting tensor shape is (nElements, dimension) instead of (nElements, 1, dimension).
In other words, if you pass only one Eigen vector, its components are treated as feature dimensions rather than as a batch of size 1. This matches the typical layout used in machine learning, where each row (or element) represents one sample, and each column represents one feature.

After adding all the blocks to the `TensorCollection`, the order of the blocks for inference can be adapted by calling `change_order()`. The order should match the expected input configuration of the PyTorch model.

More examples about usage can be found in [PyTorchAlpakaTest](../PyTorchAlpakaTest).

### Batching semantics

For batched inference, use `BatchedTensorCollection<Queue>` for the inputs and outputs. Register each SoA once with `addBatched()`, passing a batch size for tensors that should be sliced. The slice offset is computed relative to the size of that SoA, and the final batch can contain fewer elements. Omit the batch size to pass the full SoA to every batch. The `AlpakaModel::forward()` overload creates the per-batch `TensorCollection`s and runs inference for each batch.

For example, a model can receive one batch of tracks while also receiving all hits and a hit-to-track mapping, even when the track and hit SoAs have different sizes:

```cpp
BatchedTensorCollection<Queue> inputs;
BatchedTensorCollection<Queue> outputs;

inputs.addBatched<portabletest::ParticleSoA>(
    "track_features", batch_size,
    track_records.pt(), track_records.eta(), track_records.phi());
inputs.addBatched<portabletest::HitSoA>(
    "hit_features", hit_records.x(), hit_records.y(), hit_records.z());
inputs.addBatched<portabletest::HitToTrackSoA>(
    "hit_to_track", hit_to_track_records.trackIndex());
inputs.addBatched<portabletest::TrackBeginSoA>(
    "track_begin", 1, track_begin_records.trackBegin());
outputs.addBatched<portabletest::SimpleNetSoA>(
    "regression_head", batch_size,
    output_records.reco_pt());

model.forward(queue, inputs, outputs);
```

The TorchScript model is responsible for interpreting the different input sizes. In this example, all hits and the hit-to-track mapping are passed to every batch; the model uses `hit_to_track` and `track_begin` to select hits belonging to the current track batch.

All sliced SoAs, across inputs and outputs, must imply the same number of batches. Full-SoA registrations do not determine this count and are passed in full to every batch. If neither collection contains a sliced registration, inference runs once. An explicitly sliced empty SoA implies zero batches. An `SOA_SCALAR` can be registered without a batch size to pass it in full.

Full-SoA tensors share their handles across batches. For const inputs, this reuses a single writable buffer per registered full tensor; its contents are copied from the original SoA before each batch to protect the original data from model modifications. Sliced tensors retain separate per-batch handles.

Each batched `forward()` call rebuilds the materialized collections from the current registrations. The registered SoA memory must remain valid until the queued operations complete.

Full-SoA outputs refer to the same output memory for every batch. To store separate results for each batch, register the output with an explicit batch size.

For manual batching, `TensorCollection::add()` accepts a `TensorSlice{batch_id, batch_size}`. Calls without a slice expose the full SoA.

Use a positive batch size for sliced SoAs. Batch-count consistency, slice bounds, and column contiguity currently rely on `assert`.

Look at [SimpleNetMiniBatch](../PyTorchAlpakaTest/plugins/alpaka/SimpleNetMiniBatch.cc) for a batched example and [TrackHitDeepSet](../PyTorchAlpakaTest/plugins/alpaka/TrackHitDeepSet.cc) for SoAs with different sizes.

**IMPORTANT:** Choose batch sizes that respect the SoA alignment (typically a multiple of 32), otherwise an assertion can be triggered.

## FP16 Inference Support

FP16 (half precision) inference is supported alongside the default FP32 execution path. The goal is to enable reduced memory usage while preserving numerical compatibility with FP32 results.

### 1. Runtime FP16 conversion

FP32 data is stored in SoA format and explicitly converted to FP16 at inference time.
In this case, you just need to pass `torch::kHalf` to the forward call; the model and input tensors are converted to FP16 under the hood using the PyTorch API.

```cpp
// SoA with input features and output
GENERATE_SOA_LAYOUT(SimpleNetLayout, SOA_COLUMN(float, reco_pt))
GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))

TensorCollection<Queue> inputs;
inputs.add<ParticleSoA>(
    "particles",
    input_records.pt(),
    input_records.eta(),
    input_records.phi()
);
TensorCollection<Queue> outputs;
outputs.add<SimpleNetSoA>("regression_head", output_records.reco_pt());

// Runtime FP16 inference
model.forward(queue, inputs, outputs, torch::kHalf);
```

FP16 and FP32 outputs may differ slightly due to reduced precision and floating-point accumulation effects. Users are encouraged to check the output compatibility.

## Limitations
- Const correctness and thread-safety relies on `torch::from_blob()` mechanism which currently does not ensure that data will not be modified internally. There is ongoing work to support COW tensors but until this support will be integrated in mainstream PyTorch the provided solution materialises (copies) the tensors if passed registry points to `const` memory. For more information please check [Const correctness and thread-safety of torch::from_blob with external memory](https://discuss.pytorch.org/t/const-correctness-and-thread-safety-of-torch-from-blob-with-external-memory/223521) and [pytorch:#97856](https://github.com/pytorch/pytorch/issues/97856)
- For multi output branch models the intermediate copy of output is done, so there is no "true" no-copy mechanism under the hood.
- AOT support is under active development and subject to changes that obey CMSSW releasing rules.
