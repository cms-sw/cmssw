# PhysicsTools/PyTorchAlpaka
This package extends the PyTorch implementation and enables seamless integration with the Alpaka-based heterogeneous computing backend, supporting inference workflows with usage of `pytorch` library with `PortableCollection`s objects allowing users to run direct inference with reduced memory footprint. It provides:
- Compatibility with Alpaka device/queue abstractions.
- Single-threading and CUDA stream management are handled by `QueueGuard` objects specialized for each supported backend.

## Interface for Alpaka Modules
All Pytorch based modules should add `PyTorchService` to disable internal torchlib threading. It enforces single-threaded execution on CPU backends.
On GPU backends the stream managament is done with `QueueGuard<>` construct that change the Pytorch stream to the one associated with Event. Whether the PyTorch functions are used the `QueueGuard<>` has to be active and set to schedule all async operations in a CMSSW framework controlled stream.

**To enable proper execution user has to explicitly scope range of execution with `QueueGuard<Queue>` construct inside PortableModule.**

Examples demonstrating the interoperability of PyTorch with Alpaka in the CMSSW environment can be found in the [PyTorchAlpakaTest](../PyTorchAlpakaTest) directory. The basic test pipeline includes:
- *SimpleNet* composed with few Dense layers, that operate on SoA style portable data structures
- *MaskedNet* shows how to use multiple input data with `Eigen::Vector` and `SOA_SCALAR`
- *TinyResNet* emulate more complex scenario with `Eigen::Matrix` and how one can implement image-like Tensor implementation
- *MulitHeadNet* handle networks that return more than one output tensor 

## Direct Inference on SoA 
The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data (or minimal copy overhead).

### TensorRegistry
The structural information of the inputs/outputs SoA are stored in an `TensorRegistry`. Which is high level object to register recipes from which tensors are created

The `TensorRegistry` can be defined by first initializing the object and then adding recipes (`register_tensor`) to the internal metadata. Each registered tensor is transformed into a PyTorch tensor (without taking ownership) whose size and type are derived from the columns provided.

For two example SoAs Templates, which are stored in PortableCollections, columns can be added to `TensorRegistry`, by using the Metarecords implementation of SoAs.

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
PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(batch_size, queue);
fill(queue, deviceCollection);
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();
```
- **For each block** (one tensor), **add the columns** which should be merged to a single tensor. The datatypes have to be the same, and the columns have to be contiguous.

IMPORTANT: continuity of memory is strick requirement!
```
TensorRegistry input(batch_size);
input.register_tenosr<SoA>("eigen_vector", records.a(), records.b());
input.register_tenosr<SoA>("eigen_matrix", records.c());
input.register_tenosr<SoA>("column", records.x(), records.y(), records.z());
input.register_tenosr<SoA>("scalar", view.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

TensorRegistry output(batch_size);
output.register_tenosr<SoA>("result", result_view.cluster());
```

For Eigen columns, if only a single Vector/Matrix is provided for the tensor, is provided, as if each vector dimension is a column. This means size of tensor is (nElements, dimension) instead of (nElements, 1, dimension).

After adding all the blocks to the `TensorRegistry`, the order of the blocks for inference can be adapted by calling `change_order()`. The order should match the expected input configuration of the PyTorch model.

More examples about usage can be found in [PyTorchAlpakaTest](../PyTorchAlpakaTest).

## Limitations
- Current implementation supports `SerialSync` and `CudaAsync` backends only. `ROCmAsync` backend is supported via SerialSync fallback mechanism due to missing `pytorch-hip` library in CMSSW (see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75), with explicit `alpaka::wait()` call to copy data to host and back to device.
- Const correctness and thread-safety relies on `torch::from_blob()` mechanism which currently does not ensure that data will not be modified internally. There is ongoing work to support COW tensors but until this support will be integrated in mainstream PyTorch the provided solution materialize tensors if passed recipes points to `const` memory. For more information please check [Const correctness and thread-safety of torch::from_blob with external memory](https://discuss.pytorch.org/t/const-correctness-and-thread-safety-of-torch-from-blob-with-external-memory/223521) and [pytorch:#97856](https://github.com/pytorch/pytorch/issues/97856)
- For multi output branch models the intermediate copy of output is done, so there is no "true" no-copy mechanism under the hood.
- AOT support is under active development and subject to changes that obey CMSSW releasing rules.
