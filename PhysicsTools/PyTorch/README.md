## PhysicsTools/PyTorch

This package enables seamless integration between PyTorch and the Alpaka-based heterogeneous computing backend, supporting inference workflows with usage of `pytorch` library with `PortableCollection`s objects. It provides:
- Compatibility with Alpaka device/queue abstractions.
- Support for automatic conversion of optimized SoA to torch tensors, with memory blobs reusage.
- Support for both just-in-time (JIT) and ahead-of-time (AOT) model execution (Beta version for AOT).
- Single-threading and CUDA stream management are handled by Guard objects specialized for each supported backend.

### Alpaka Config
To enable alpaka aware compilation the config files specify the appropriate mappings between common structures like device types:
```cpp
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CUDA;
// #elif ALPAKA_ACC_GPU_HIP_ENABLED  // not supported
// constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif
```

### Interface for Alpaka Modules
PyTorch internally uses optimizations such as a custom thread pool initialized with all available threads by default. To manage thread usage and control execution context, a Guard object is provided. It enforces single-threaded execution on CPU backends and sets the appropriate stream (e.g., CUDA stream) on GPU backends based on the associated Queue event.
> To enable proper execution user has to explicitly scope range of execution with `Guard<Queue>` construct inside Module.

Examples demonstrating the interoperability of PyTorch with Alpaka in the CMSSW environment can be found in the `plugins` directory. The basic test pipeline includes one input data producer and three heterogeneous modules: a dummy Alpaka module with a basic kernel, a second module performing machine learning inference on the target backend, and a third module explicitly configured to run inference on the CPU backend. This setup serves as a test case for evaluating behavior in a multithreaded and multistream CMSSW environment where inference is executed in parallel across different devices.
### Inference: JIT and AOT Model Execution
Interface provide `Model` class which is templated over compilation type (possible choices: `kAheadOfTime`, `kJustInTime`) providing wrapper enabling inference with SoA objects by incorporating necessary conversions with reduced number of memcpy operations.

#### Just-in-Time:
- Loads `torch::jit::script::Module` at runtime.
- Compiles model on-the-fly.
- Introduces warm-up overhead without additional optimization.

```cpp
auto m_path = get_path("example_model.pt");
Model<CompilationType::kJustInTime> jit_model(m_path);
jit_model.to(queue);
CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == jit_model.device());
auto outputs = jit_model.forward(inputs);
```

#### Ahead-of-Time (beta version):
- Uses PyTorch AOT compiler to generate `.cpp` and `.so` files. (prerequisite done manually by end-user)
    - Package provide helper script to automate this process to some extent `PhysicsTools/PyTorch/scripts/aot.sh` (run from within `PhysicsTools/PyTorch` directory)
- Loads compiled model via `AOTIModelContainerRunner`.
- Eliminates JIT overhead, enable optimization, but requires architecture-specific handling.

```cpp
auto lib_path = shared_lib();
auto m_path = get_path("example_precompiled_model.pt2");
Model<CompilationType::kAheadOfTime> aot_model(m_path);
aot_model.to(queue); 
CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == aot_model.device());
auto outputs = aot_model.forward(inputs_tensor);
```


### PyTorch Wrapper for C++ and Alpaka

The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data. This can be used directly with a PyTorch model. The result can also be dynamically placed into a SoA buffer.

#### Metadata

The structual information of the input and output SoA are stored in an `SoAMetadata`. These two objects are then combined to a `ModelMetadata`, to be used by the `Converter`.

#### Defining Metadata

The `SoAMetadata' can be defined by first initialising the object and then adding blocks to the metadata. Each block is transformed into a tensor whose size and type are derived from the columns provided.

Example SOA Template for Model Input:
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

Example SOA Template for Model Output:
```cpp
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster));
```

#### Metadata Definition (Automatic Approach):
```cpp
PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(batch_size, queue);
fill(queue, deviceCollection);
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();

SoAMetadata<SoA> input(batch_size);
input.append_block("eigen_vector", records.a(), records.b());
input.append_block("eigen_matrix", records.c());
input.append_block("column", records.x(), records.y(), records.z());
input.append_block("scalar", view.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

SoAMetadata<SoA> output(batch_size);
output.append_block("result", result_view.cluster());
ModelMetadata metadata(input, output);
```

Example usage with model:
```cpp
// alpaka setup
Platform platform;
std::vector<Device> alpaka_devices = alpaka::getDevs(platform);
const auto& alpaka_host = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
CPPUNIT_ASSERT(alpaka_devices.size());
const auto& alpaka_device = alpaka_devices[0];
Queue queue{alpaka_device};

const std::size_t batch_size = 32;

// host structs
PortableHostCollection<SoAInputs> inputs_host(batch_size, cms::alpakatools::host());
PortableHostCollection<SoAOutputs> outputs_host(batch_size, cms::alpakatools::host());
// device structs
PortableCollection<SoAInputs, Device> inputs_device(batch_size, alpaka_device);
PortableCollection<SoAOutputs, Device> outputs_device(batch_size, alpaka_device);

// prepare inputs
for (size_t i = 0; i < batch_size; i++) { 
    inputs_host.view().x()[i] = 0.0f;
    inputs_host.view().y()[i] = 0.0f;
    inputs_host.view().z()[i] = 0.0f;
}
alpaka::memcpy(queue, inputs_device.buffer(), inputs_host.buffer());
alpaka::wait(queue);

{
    // guard scope
    cms::torch::alpaka::set_threading_guard();
    cms::torch::alpaka::Guard<Queue> guard(queue);  

    // instantiate model
    auto m_path = get_path("example_model.pt");
    auto model = cms::torch::alpaka::Model<cms::torch::alpaka::CompilationType::kJustInTime>(m_path);
    model.to(queue);
    CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == model.device());

    // metadata for automatic tensor conversion
    auto input_records = inputs_device.view().records();
    auto output_records = outputs_device.view().records();
    cms::torch::alpaka::SoAMetadata<SoAInputs> inputs_metadata(batch_size);
    inputs_metadata.append_block("features", input_records.x(), input_records.y(), input_records.z());
    cms::torch::alpaka::SoAMetadata<SoAOutputs> outputs_metadata(batch_size); 
    outputs_metadata.append_block("preds", output_records.m(), output_records.n());
    cms::torch::alpaka::ModelMetadata<SoAInputs, SoAOutputs> metadata(inputs_metadata, outputs_metadata);
    
    // inference
    model.forward(metadata);

    // check outputs
    alpaka::memcpy(queue, outputs_host.buffer(), outputs_device.buffer());
    alpaka::wait(queue);
    for (size_t i = 0; i < batch_size; i++) {
        CPPUNIT_ASSERT(outputs_host.const_view().m()[i] == 0.5f);
        CPPUNIT_ASSERT(outputs_host.const_view().n()[i] == 0.5f);
    }
}
```

#### Ordering of Blocks

The function `change_order()` in the allows specifying the order in which the blocks should be processed. The order should match the expected input configuration of the PyTorch model.



### Limitations
- Current implementation supports CUDA backend only. ROCm backend is not yet supported, see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75ROCm.  Connected with:
    - #9786 Pytorch with ROCm (temp): https://github.com/cms-sw/cmsdist/pull/9312
    - #9312 [WIP] Build PyTorch with ROCm https://github.com/cms-sw/cmsdist/pull/9786
- AOT support is under active development and subject to changes that obey CMSSW releasing rules.
- On more complex models with multiple output branches extra copy is needed 