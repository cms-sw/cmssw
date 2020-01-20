# CUDA algorithms in CMSSW

## Outline

* [Introduction](#introduction)
  * [Design goals](#design-goals)
  * [Overall guidelines](#overall-guidelines)
* [Sub-packages](#sub-packages)
* [Examples](#examples)
  * [Isolated producer (no CUDA input nor output)](#isolated-producer-no-cuda-input-nor-output)
  * [Producer with CUDA output](#producer-with-cuda-output)
  * [Producer with CUDA input](#producer-with-cuda-input)
  * [Producer with CUDA input and output (with ExternalWork)](#producer-with-cuda-input-and-output-with-externalwork)
  * [Producer with CUDA input and output, and internal chain of CPU and GPU tasks (with ExternalWork)](producer-with-cuda-input-and-output-and-internal-chain-of-cpu-and-gpu-tasks-with-externalwork)
  * [Producer with CUDA input and output (without ExternalWork)](#producer-with-cuda-input-and-output-without-externalwork)
  * [Analyzer with CUDA input](#analyzer-with-cuda-input)
  * [Configuration](#configuration)
    * [GPU-only configuration](#gpu-only-configuration)
    * [Automatic switching between CPU and GPU modules](#automatic-switching-between-cpu-and-gpu-modules)
* [More details](#more-details)
  * [Device choice](#device-choice)
  * [Data model](#data-model)
  * [CUDA EDProducer](#cuda-edproducer)
    * [Class declaration](#class-declaration)
    * [Memory allocation](#memory-allocation)
      * [Caching allocator](#caching-allocator)
      * [Non-cached pinned host `unique_ptr`](#non-cached-pinned-host-unique_ptr)
      * [CUDA API](#cuda-api)
    * [Setting the current device](#setting-the-current-device)
    * [Getting input](#getting-input)
    * [Calling the CUDA kernels](#calling-the-cuda-kernels)
    * [Putting output](#putting-output)
    * [`ExternalWork` extension](#externalwork-extension)
    * [Module-internal chain of CPU and GPU tasks](#module-internal-chain-of-cpu-and-gpu-tasks)
    * [Transferring GPU data to CPU](#transferring-gpu-data-to-cpu)
    * [Synchronizing between CUDA streams](#synchronizing-between-cuda-streams)
  * [CUDA ESProduct](#cuda-esproduct)

## Introduction

This page documents the CUDA integration within CMSSW

### Design goals

1. Provide a mechanism for a chain of modules to share a resource
   * Resource can be e.g. CUDA device memory or a CUDA stream
2. Minimize data movements between the CPU and the device
3. Support multiple devices
4. Allow the same job configuration to be used on all hardware combinations

### Overall guidelines

1. Within the `acquire()`/`produce()` functions all CUDA operations should be asynchronous, i.e.
   * Use `cudaMemcpyAsync()`, `cudaMemsetAsync()`, `cudaMemPrefetchAsync()` etc.
   * Avoid `cudaMalloc*()`, `cudaHostAlloc()`, `cudaFree*()`, `cudaHostRegister()`, `cudaHostUnregister()` on every event
     * Occasional calls are permitted through a caching mechanism that amortizes the cost (see also [Caching allocator](#caching-allocator))
   * Avoid `assert()` in device functions, or use `#include HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h`
     * With the latter the `assert()` calls in CUDA code are disabled by
       default, but can be enabled by defining a `GPU_DEBUG` macro
       (before the aforementioned include)
2. Synchronization needs should be fulfilled with
   [`ExternalWork`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#edm_ExternalWork)
   extension to EDProducers
   * `ExternalWork` can be used to replace one synchronization point
     (e.g. between device kernels and copying a known amount of data
     back to CPU).
   * For further synchronization points (e.g. copying data whose
     amount is known only at the device side), split the work to
     multiple `ExternalWork` producers. This approach has the added
     benefit that e.g. data transfers to CPU become on-demand automatically
   * A general breakdown of the possible steps:
     * Convert input legacy CPU data format to CPU SoA
     * Transfer input CPU SoA to GPU
     * Launch kernels
     * Transfer the number of output elements to CPU
     * Transfer the output data from GPU to CPU SoA
     * Convert the output SoA to legacy CPU data formats
3. Within `acquire()`/`produce()`, the current CUDA device is set
   implicitly and the CUDA stream is provided by the system (with
   `cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`)
   * It is strongly recommended to use the provided CUDA stream for all operations
     * If that is not feasible for some reason, the provided CUDA
       stream must synchronize with the work queued on other CUDA
       streams (with CUDA events and `cudaStreamWaitEvent()`)
4. Outside of `acquire()`/`produce()`, CUDA API functions may be
   called only if `CUDAService::enabled()` returns `true`.
   * With point 3 it follows that in these cases multiple devices have
     to be dealt with explicitly, as well as CUDA streams

## Sub-packages
* [`HeterogeneousCore/CUDACore`](#cuda-integration) CUDA-specific core components
* [`HeterogeneousCore/CUDAServices`](../CUDAServices) Various edm::Services related to CUDA
* [`HeterogeneousCore/CUDAUtilities`](../CUDAUtilities) Various utilities for CUDA kernel code
* [`HeterogeneousCore/CUDATest`](../CUDATest) Test modules and configurations
* [`CUDADataFormats/Common`](../../CUDADataFormats/Common) Utilities for event products with CUDA data

## Examples

### Isolated producer (no CUDA input nor output)

```cpp
class IsolatedProducerCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  ...
private:
  ...
  IsolatedProducerGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<InputData> inputToken_;
  edm::EDPutTokenT<OutputData> outputToken_;
};
...
void IsolatedProducerCUDA::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Sets the current device and creates a CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  auto const& inputData = iEvent.get(inputToken_);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAcquire::stream()
  gpuAlgo_.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void IsolatedProducerCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // Real life is likely more complex than this simple example. Here
  // getResult() returns some data in CPU memory that is passed
  // directly to the OutputData constructor.
  iEvent.emplace(outputToken_, gpuAlgo_.getResult());
}
```

### Producer with CUDA output

```cpp
class ProducerOutputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  ...
private:
  ...
  ProducerOutputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<InputData> inputToken_;
  edm::EDPutTokenT<cms::cuda::Product<OutputData>> outputToken_;
  cms::cuda::ContextState ctxState_;
};
...
void ProducerOutputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Sets the current device and creates a CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  auto const& inputData = iEvent.get(inputToken_);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAcquire::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished,
  // and saves the device and CUDA stream to ctxState_
}

// Called after the asynchronous work has finished
void ProducerOutputCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // Sets again the current device, uses the CUDA stream created in the acquire()
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  // Now getResult() returns data in GPU memory that is passed to the
  // constructor of OutputData. cms::cuda::ScopedContextProduce::emplace() wraps the
  // OutputData to cms::cuda::Product<OutputData>. cms::cuda::Product<T> stores also
  // the current device and the CUDA stream since those will be needed
  // in the consumer side.
  ctx.emplace(iEvent, outputToken_, gpuAlgo.getResult());
}
```

### Producer with CUDA input

```cpp
class ProducerInputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<cms::cuda:Product<InputData>> inputToken_;
  edm::EDGetTokenT<cms::cuda::Product<OtherInputData>> otherInputToken_;
  edm::EDPutTokenT<OutputData> outputToken_;
};
...
void ProducerInputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and possibly use the same CUDA stream
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the cms::cuda::ScopedContextAcquire holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Input data from another producer
  auto const& otherInputData = ctx.get(iEvent.get(otherInputToken_));
  // or
  auto const& otherInputData = ctx.get(iEvent, otherInputToken_);


  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAcquire::stream()
  gpuAlgo.makeAsync(inputData, otherInputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void ProducerInputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Real life is likely more complex than this simple example. Here
  // getResult() returns some data in CPU memory that is passed
  // directly to the OutputData constructor.
  iEvent.emplace(outputToken_, gpuAlgo_.getResult());
}
```

See [further below](#setting-the-current-device) for the conditions
when the `cms::cuda::ScopedContextAcquire` constructor reuses the CUDA stream. Note
that the `cms::cuda::ScopedContextAcquire` constructor taking `edm::StreamID` is
allowed, it will just always create a new CUDA stream.


### Producer with CUDA input and output (with ExternalWork)

```cpp
class ProducerInputOutputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<cms::cuda::Product<InputData>> inputToken_;
  edm::EDPutTokenT<cms::cuda::Product<OutputData>> outputToken_;
};
...
void ProducerInputOutputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and also use the same CUDA stream
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder), ctxState_};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the cms::cuda::ScopedContextAcquire holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAcquire::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished,
  // and saves the device and CUDA stream to ctxState_
}

// Called after the asynchronous work has finished
void ProducerInputOutputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Sets again the current device, uses the CUDA stream created in the acquire()
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  // Now getResult() returns data in GPU memory that is passed to the
  // constructor of OutputData. cms::cuda::ScopedContextProduce::emplace() wraps the
  // OutputData to cms::cuda::Product<OutputData>. cms::cuda::Product<T> stores also
  // the current device and the CUDA stream since those will be needed
  // in the consumer side.
  ctx.emplace(iEvent, outputToken_, gpuAlgo.getResult());
}
```

[Complete example](../CUDATest/plugins/TestCUDAProducerGPUEW.cc)


### Producer with CUDA input and output, and internal chain of CPU and GPU tasks (with ExternalWork)

```cpp
class ProducerInputOutputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  void addMoreWork(edm::WaitingTaskWithArenaHolder waitingTashHolder);

  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<cms::cuda::Product<InputData>> inputToken_;
  edm::EDPutTokenT<cms::cuda::Product<OutputData>> outputToken_;
};
...
void ProducerInputOutputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and also use the same CUDA stream
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder), ctxState_};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the cms::cuda::ScopedContextAcquire holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAcquire::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Push a functor on top of "a stack of tasks" to be run as a next
  // task after the work queued above before produce(). In this case ctx
  // is a context constructed by the calling TBB task, and therefore the
  // current device and CUDA stream have been already set up. The ctx
  // internally holds the WaitingTaskWithArenaHolder for the next task.

  ctx.pushNextTask([this](cms::cuda::ScopedContextTask ctx) {
    addMoreWork(ctx);
  });

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished,
  // and saves the device and CUDA stream to ctxState_
}

// Called after the asynchronous work queued in acquire() has finished
void ProducerInputOutputCUDA::addMoreWork(cms::cuda::ScopedContextTask& ctx) {
  // Current device and CUDA stream have already been set

  // Queues more asynchronous data transfer and kernels to the CUDA
  // stream returned by cms::cuda::ScopedContextTask::stream()
  gpuAlgo.makeMoreAsync(ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work queued in addMoreWork() has finished
void ProducerInputOutputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Sets again the current device, uses the CUDA stream created in the acquire()
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  // Now getResult() returns data in GPU memory that is passed to the
  // constructor of OutputData. cms::cuda::ScopedContextProduce::emplace() wraps the
  // OutputData to cms::cuda::Product<OutputData>. cms::cuda::Product<T> stores also
  // the current device and the CUDA stream since those will be needed
  // in the consumer side.
  ctx.emplace(iEvent, outputToken_, gpuAlgo.getResult());
}
```

[Complete example](../CUDATest/plugins/TestCUDAProducerGPUEWTask.cc)


### Producer with CUDA input and output (without ExternalWork)

If the producer does not need to transfer anything back to CPU (like
the number of output elements), the `ExternalWork` extension is not
needed as there is no need to synchronize.

```cpp
class ProducerInputOutputCUDA: public edm::global::EDProducer<> {
public:
  ...
  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup& iSetup) const override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<cms::cuda::Product<InputData>> inputToken_;
  edm::EDPutTokenT<cms::cuda::Product<OutputData>> outputToken_;
};
...
void ProducerInputOutputCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup& iSetup) const {
  cms::cuda::Product<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and possibly use the same CUDA stream
  cms::cuda::ScopedContextProduce ctx{inputDataWrapped};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the cms::cuda::ScopedContextProduce holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextProduce::stream(). Here makeAsync() also
  // returns data in GPU memory that is passed to the constructor of
  // OutputData. cms::cuda::ScopedContextProduce::emplace() wraps the OutputData to
  // cms::cuda::Product<OutputData>. cms::cuda::Product<T> stores also the current
  // device and the CUDA stream since those will be needed in the
  // consumer side.
  ctx.emplace(iEvent, outputToken, gpuAlgo.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}
```

[Complete example](../CUDATest/plugins/TestCUDAProducerGPU.cc)


### Analyzer with CUDA input

Analyzer with CUDA input is similar to [producer with CUDA
input](#producer-with-cuda-input). Note that currently we do not have
a mechanism for portable configurations with analyzers (like
[`SwitchProducer`](#automatic-switching-between-cpu-and-gpu-modules)
for producers). This means that a configuration with a CUDA analyzer
can only run on a machine with CUDA device(s).

```cpp
class AnalyzerInputCUDA: public edm::global::EDAnalyzer<> {
public:
  ...
  void analyzer(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  ...
private:
  ...
  AnalyzerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<cms::cuda::Product<InputData>> inputToken_;
  edm::EDGetTokenT<cms::cuda::Product<OtherInputData>> otherInputToken_;
};
...
void AnalyzerInputCUDA::analyze(edm::Event const& iEvent, edm::EventSetup& iSetup) {
  cms::cuda::Product<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and possibly use the same CUDA stream
  cms::cuda::ScopedContextAnalyze ctx{inputDataWrapped};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the cms::cuda::ScopedContextAnalyze holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Input data from another producer
  auto const& otherInputData = ctx.get(iEvent.get(otherInputToken_));
  // or
  auto const& otherInputData = ctx.get(iEvent, otherInputToken_);


  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by cms::cuda::ScopedContextAnalyze::stream()
  gpuAlgo.analyzeAsync(inputData, otherInputData, ctx.stream());
}
```

[Complete example](../CUDATest/plugins/TestCUDAAnalyzerGPU.cc)


### Configuration

#### GPU-only configuration

For a GPU-only configuration there is nothing special to be done, just
construct the Paths/Sequences/Tasks from the GPU modules.

#### Automatic switching between CPU and GPU modules

The `SwitchProducer` mechanism can be used to switch automatically
between CPU and GPU modules based on the availability of GPUs on the
machine where the configuration is done. Framework decides at the
beginning of the job which of the modules to run for a given module
label.

Framework requires that the modules in the switch must produce the
same types of output products (the closer the actual results are the
better, but the framework can not enforce that). This means that for a
chain of GPU modules, it is the module that transforms the SoA data
format back to the legacy data formats (possibly, but not necessarily,
transferring the SoA data from GPU to CPU) that should be switched
between the legacy CPU module. The rest of the GPU modules should be
placed to a `Task`, in which case framework runs them only if their
output is needed by another module.

```python
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
process.foo = SwitchProducerCUDA(
    cpu = cms.EDProducer("FooProducer"), # legacy CPU
    cuda = cms.EDProducer("FooProducerFromCUDA",
        src="fooCUDA"
    )
)
process.fooCUDA = cms.EDProducer("FooProducerCUDA")

process.fooTaskCUDA = cms.Task(process.fooCUDA)
process.fooTask = cms.Task(
    process.foo,
    process.fooTaskCUDA
)
```

For a more complete example, see [here](../CUDATest/test/testCUDASwitch_cfg.py).





## More details

### Device choice

As discussed above, with `SwitchProducer` the choice between CPU and
GPU modules is done at the beginning of the job.

For multi-GPU setup the device is chosen in the first CUDA module in a
chain of modules by one of the constructors of
`cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`
```cpp
// In ExternalWork acquire()
cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), ...};

// In normal produce() (or filter())
cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};
```
As the choice is still the static EDM stream to device assignment, the
EDM stream ID is needed. The logic will likely evolve in the future to
be more dynamic, and likely the device choice has to be made for the
full event.

### Data model

The "GPU data product" should be a class/struct containing smart
pointer(s) to device data (see [Memory allocation](#memory-allocation)).
When putting the data to event, the data is wrapped to
`cms::cuda::Product<T>` template, which holds
* the GPU data product
  * must be moveable, but no other restrictions
* the current device where the data was produced, and the CUDA stream the data was produced with
* [CUDA event for synchronization between multiple CUDA streams](#synchronizing-between-cuda-streams)

Note that the `cms::cuda::Product<T>` wrapper can be constructed only with
`cms::cuda::ScopedContextProduce::wrap()`, and the data `T` can be obtained
from it only with
`cms::cuda::ScopedContextAcquire::get()`/`cms::cuda::ScopedContextProduce::get()`/`cms::cuda::ScopedContextAnalyze::get()`,
as described further below. When putting the data product directly to
`edm::Event`, also `cms::cuda::SCopedContextProduce::emplace()` can be used.

The GPU data products that depend on the CUDA runtime should be placed
under `CUDADataFormats` package, using the same name for sub-package
that would be used in `DataFormats`. Everything else, e.g. SoA for
CPU, should go under `DataFormats` as usual.


### CUDA EDProducer

#### Class declaration

The CUDA producers are normal EDProducers. The `ExternalWork`
extension should be used if a synchronization between the GPU and CPU
is needed, e.g. when transferring data from GPU to CPU.

#### Memory allocation

##### Caching allocator

The memory allocations should be done dynamically with the following functions
```cpp
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> device_buffer = cms::cuda::make_device_unique<float[]>(50, cudaStream);
cms::cuda::host::unique_ptr<float[]>   host_buffer   = cms::cuda::make_host_unique<float[]>(50, cudaStream);
```

in the `acquire()` and `produce()` functions. The same
`cudaStream_t` object that is used for transfers and kernels
should be passed to the allocator.

The allocator is based on `cub::CachingDeviceAllocator`. The memory is
guaranteed to be reserved
* for the host: up to the destructor of the `unique_ptr`
* for the device: until all work queued in the `cudaStream` up to the point when the `unique_ptr` destructor is called has finished

##### Non-cached pinned host `unique_ptr`

In producers transferring data to GPU one may want to pinned host
memory allocated with `cudaHostAllocWriteCombined`. As of now we don't
want to include the flag dimension to the caching allocator. The CUDA
API wrapper library does not support allocation flags, so we add our
own `unique_ptr` for that.

```cpp
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

cms::cuda::host::noncached_unique_ptr<float[]> host_buffer = cms::cuda::make_host_noncached_unique<float[]>(50, flags);
```
The `flags` is passed directly to `cudaHostAlloc()`.

##### CUDA API

The `cudaMalloc()` etc may be used outside of the event loop, but that
should be limited to only relatively small allocations in order to
allow as much re-use of device memory as possible.

If really needed, the `cudaMalloc()` etc may be used also within the
event loop, but then the cost of allocation and implicit
synchronization should be explicitly amortized e.g. by caching.

#### Setting the current device

A CUDA producer should construct `cms::cuda::ScopedContextAcquire` in
`acquire()` (`cms::cuda::ScopedContextProduce` `produce()` if not using
`ExternalWork`) either with `edm::StreamID`, or with a
`cms::cuda::Product<T>` read as an input.

```cpp
// From edm::StreamID
cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), ...};
// or
cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};


// From cms::cuda::Product<T>
cms::cuda::Product<GPUClusters> const& cclus = iEvent.get(srcToken_);
cms::cuda::ScopedContextAcquire ctx{cclus, ...};
// or
cms::cuda::ScopedContextProduce ctx{cclus};
```

A CUDA analyzer should construct `cms::cuda::ScopedContextAnalyze` with a
`cms::cuda::Product<T>` read as an input.

```cpp
cms::cuda::Product<GPUClusters> const& cclus = iEvent.get(srcToken_);
cms::cuda::ScopedContextAnalyze ctx{cclus};
```

`cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`/`cms::cuda::ScopedContextAnalyze` work in the RAII way and does the following
* Sets the current device for the current scope
  - If constructed from the `edm::StreamID`, chooses the device and creates a new CUDA stream
  - If constructed from the `cms::cuda::Product<T>`, uses the same device and possibly the same CUDA stream as was used to produce the `cms::cuda::Product<T>`
    * The CUDA stream is reused if this producer is the first consumer
      of the `cms::cuda::Product<T>`, otherwise a new CUDA stream is created.
      This approach is simple compromise to automatically express the work of
      parallel producers in different CUDA streams, and at the same
      time allow a chain of producers to queue their work to the same
      CUDA stream.
* Gives access to the CUDA stream the algorithm should use to queue asynchronous work
* `cms::cuda::ScopedContextAcquire` calls `edm::WaitingTaskWithArenaHolder::doneWaiting()` when necessary (in its destructor)
* [Synchronizes between CUDA streams if necessary](#synchronizing-between-cuda-streams)
* Needed to get `cms::cuda::Product<T>` from the event
  * `cms::cuda::ScopedContextProduce` is needed to put `cms::cuda::Product<T>` to the event

In case of multiple input products, from possibly different CUDA
streams and/or CUDA devices, this approach gives the developer full
control in which of them the kernels of the algorithm should be run.

#### Getting input

The real product (`T`) can be obtained from `cms::cuda::Product<T>` only with
the help of
`cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`/`cms::cuda::ScopedContextAnalyze`.

```cpp
// From cms::cuda::Product<T>
cms::cuda::Product<GPUClusters> cclus = iEvent.get(srcToken_);
GPUClusters const& clus = ctx.get(cclus);

// Directly from Event
GPUClusters const& clus = ctx.get(iEvent, srcToken_);
```

This step is needed to
* check that the data are on the same CUDA device
  * if not, throw an exception (with unified memory could prefetch instead)
* if the CUDA streams are different, synchronize between them

#### Calling the CUDA kernels

It is usually best to wrap the CUDA kernel calls to a separate class,
and then call methods of that class from the EDProducer. The only
requirement is that the CUDA stream where to queue the operations
should be the one from the
`cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`/`cms::cuda::ScopedContextAnalyze`.

```cpp
gpuAlgo.makeClustersAsync(..., ctx.stream());
```

If necessary, different CUDA streams may be used internally, but they
should to be made to synchronize with the provided CUDA stream with
CUDA events and `cudaStreamWaitEvent()`.


#### Putting output

The GPU data needs to be wrapped to `cms::cuda::Product<T>` template with
`cms::cuda::ScopedContextProduce::wrap()` or `cms::cuda::ScopedContextProduce::emplace()`

```cpp
GPUClusters clusters = gpuAlgo.makeClustersAsync(..., ctx.stream());
std::unique_ptr<cms::cuda::Product<GPUClusters>> ret = ctx.wrap(clusters);
iEvent.put(std::move(ret));

// or with one line
iEvent.put(ctx.wrap(gpuAlgo.makeClustersAsync(ctx.stream())));

// or avoid one unique_ptr with emplace
edm::PutTokenT<cms::cuda::Product<GPUClusters>> putToken_ = produces<cms::cuda::Product<GPUClusters>>(); // in constructor
...
ctx.emplace(iEvent, putToken_, gpuAlgo.makeClustersAsync(ctx.stream()));
```

This step is needed to
* store the current device and CUDA stream into `cms::cuda::Product<T>`
* record the CUDA event needed for CUDA stream synchronization

#### `ExternalWork` extension

Everything above works both with and without `ExternalWork`.

Without `ExternalWork` the `EDProducer`s act similar to TBB
flowgraph's "streaming node". In other words, they just queue more
asynchronous work to the CUDA stream in their `produce()`.

The `ExternalWork` is needed when one would otherwise call
`cudeStreamSynchronize()`. For example transferring something to CPU
needed for downstream DQM, or queueing more asynchronous work. With
`ExternalWork` an `acquire()` method needs to be implemented that gets
an `edm::WaitingTaskWithArenaHolder` parameter. The
`edm::WaitingTaskWithArenaHolder` should then be passed to the
constructor of `cms::cuda::ScopedContextAcquire` along

```cpp
void acquire(..., edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<GPUClusters> const& cclus = iEvent.get(token_);
  cms::cuda::ScopedContextAcquire ctx{cclus, std::move(waitingTaskHolder)}; // can also copy instead of move if waitingTaskHolder is needed for something else as well
  ...
```

When constructed this way, `cms::cuda::ScopedContextAcquire` registers a
callback function to the CUDA stream in its destructor to call
`waitingTaskHolder.doneWaiting()`.

A GPU->GPU producer needs a `cms::cuda::ScopedContext` also in its
`produce()`. The device and CUDA stream are transferred via
`cms::cuda::ContextState` member variable:

```cpp
class FooProducerCUDA ... {
  ...
  cms::cuda::ContextState ctxState_;
};

void FooProducerCUDA::acquire(...) {
  ...
  cms::cuda::ScopedContextAcquire ctx{..., std::move(waitingTaskHolder), ctxState_};
  ...
}

void FooProducerCUDA::produce(...( {
  ...
  cms::cuda::ScopedContextProduce ctx{ctxState_};
}
```

The `cms::cuda::ScopedContextAcquire` saves its state to the `ctxState_` in
the destructor, and `cms::cuda::ScopedContextProduce` then restores the
context.

#### Module-internal chain of CPU and GPU tasks

Technically `ExternalWork` works such that the framework calls
`acquire()` with a `edm::WaitingTaskWithArenaHolder` that holds an
`edm::WaitingTask` (that inherits from `tbb::task`) for calling
`produce()` in a `std::shared_ptr` semantics: spawn the task when
reference count hits `0`. It is also possible to create a longer chain
of such tasks, alternating between CPU and GPU work. This mechanism
can also be used to re-run (part of) the GPU work.

The "next tasks" to run are essentially structured as a stack, such
that
- `cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextTask::pushNextTask()`
  pushes a new functor on top of the stack
- Completion of both the asynchronous work and the queueing function
  pops the top task of the stack and enqueues it (so that TBB
  eventually runs the task)
  * Technically the task is made eligible to run when all copies of
    `edm::WaitingTaskWithArenaHolder` of the acquire() (or "previous"
    function) have either been destructed or their `doneWaiting()` has
    been called
  * The code calling `acquire()` or the functor holds one copy of
    `edm::WaitingTaskWithArenaHolder` so it is guaranteed that the
    next function will not run before the earlier one has finished


Below is an example how to push a functor on top of the stack of tasks
to run next (following the example of the previous section)
```cpp
void FooProducerCUDA::acquire(...) {
   ...
   ctx.pushNextTask([this](cms::cuda::ScopedContextTask ctx) {
     ...
   });
   ...
}
```

In this case the `ctx`argument to the function is a
`cms::cuda::ScopedContexTask` object constructed by the TBB task calling the
user-given function. It follows that the current device and CUDA
stream have been set up already. The `pushNextTask()` can be called
many times. On each invocation the `pushNextTask()` pushes a new task
on top of the stack (i.e. in front of the chain). It follows that in
```cpp
void FooProducerCUDA::acquire(...) {
   ...
   ctx.pushNextTask([this](cms::cuda::ScopedContextTask ctx) {
     ... // function 1
   });
   ctx.pushNextTask([this](cms::cuda::ScopedContextTask ctx) {
     ... // function 2
   });
   ctx.pushNextTask([this](cms::cuda::ScopedContextTask ctx) {
     ... // function 3
   });
   ...
}
```
the functions will be run in the order 3, 2, 1.

**Note** that the `CUDAService` is **not** available (nor is any other
service) in these intermediate tasks. In the near future memory
allocations etc. will be made possible by taking them out from the
`CUDAService`.

The `cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextTask` have also a
more generic member function, `replaceWaitingTaskHolder()`, that can
be used to just replace the currently-hold
`edm::WaitingTaskWithArenaHolder` (that will get notified by the
callback function) with anything. In this case the caller is
responsible of creating the task(s) and setting up the chain of them.


#### Transferring GPU data to CPU

The GPU->CPU data transfer needs synchronization to ensure the CPU
memory to have all data before putting that to the event. This means
the `ExternalWork` needs to be used along
* In `acquire()`
  * (allocate CPU memory buffers)
  * Queue all GPU->CPU transfers asynchronously
* In `produce()`
  * If needed, read additional CPU products (e.g. from `edm::Ref`s)
  * Reformat data back to legacy data formats
  * Note: `cms::cuda::ScopedContextProduce` is **not** needed in `produce()`

#### Synchronizing between CUDA streams

In case the producer needs input data that were produced in two (or
more) CUDA streams, these streams have to be synchronized. Here this
synchronization is achieved with CUDA events.

Each `cms::cuda::Product<T>` constains also a CUDA event object. The call to
`cms::cuda::ScopedContextProduce::wrap()` will *record* the event in the CUDA
stream. This means that when all work queued to the CUDA stream up to
that point has been finished, the CUDA event becomes *occurred*. Then,
in
`cms::cuda::ScopedContextAcquire::get()`/`cms::cuda::ScopedContextProduce::get()`/`cms::cuda::ScopedContextAnalyze::get()`,
if the `cms::cuda::Product<T>` to get from has a different CUDA stream than
the
`cms::cuda::ScopedContextAcquire`/`cms::cuda::ScopedContextProduce`/`cms::cuda::ScopedContextAnalyze`,
`cudaStreamWaitEvent(stream, event)` is called. This means that all
subsequent work queued to the CUDA stream will wait for the CUDA event
to become occurred. Therefore this subsequent work can assume that the
to-be-getted CUDA product exists.


### CUDA ESProduct

Conditions data can be transferred to the device with the following
pattern.

1. Define a `class`/`struct` for the data to be transferred in the format accessed in the device (hereafter referred to as "payload")
2. Define a wrapper ESProduct that holds the aforementioned data in the pinned host memory
3. The wrapper should have a function returning the payload on the
   device memory. The function should transfer the data to the device
   asynchronously with the help of `cms::cuda::ESProduct<T>`.

#### Example

```cpp
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"

// Declare the struct for the payload to be transferred. Here the
// example is an array with (potentially) dynamic size. Note that all of
// below becomes simpler if the array has compile-time size.
struct ESProductExampleCUDA {
  float *someData;
  unsigned int size;
};

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class ESProductExampleCUDAWrapper {
public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  ESProductExampleCUDAWrapper(ESProductExample const&);

  // Deallocates all pinned host memory
  ~ESProductExampleCUDAWrapper();

  // Function to return the actual payload on the memory of the current device
  ESProductExampleCUDA const *getGPUProductAsync(cudaStream_t stream) const;

private:
  // Holds the data in pinned CPU memory
  float *someData_;
  unsigned int size_;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    // internal pointers are on device, struct itself is on CPU
    ESProductExampleCUDA *esproductHost = nullptr;
    // internal pounters and struct are on device
    ESProductExampleCUDA *esproductDevice = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

ESProductExampleCUDAWrapper::ESProductExampleCUDAWrapper(ESProductExample const& cpuProduct) {
  cudaCheck(cudaMallocHost(&someData_, sizeof(float)*NUM_ELEMENTS));
  // fill someData_ and size_ from cpuProduct
}

ESProductExampleCUDA const *ESProductExampleCUDAWrapper::getGPUProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate memory. Currently this can be with the CUDA API,
    // sometime we'll migrate to the caching allocator. Assumption is
    // that IOV changes are rare enough that adding global synchronization
    // points is not that bad (for now).

    // Allocate the payload object on pinned host memory.
    cudaCheck(cudaMallocHost(&data.esproductHost, sizeof(ESProductExampleCUDA)));
    // Allocate the payload array(s) on device memory.
    cudaCheck(cudaMalloc(&data.esproductHost->someData, sizeof(float)*NUM_ELEMENTS));

    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.esproductDevice, sizeof(ESProductDevice)));

    // Complete the host-side information on the payload
    data.cablingMapHost->size = this->size_;


    // Transfer the payload, first the array(s) ...
    cudaCheck(cudaMemcpyAsync(data.esproductHost->someData, this->someData, sizeof(float)*NUM_ELEMENTS, cudaMemcpyDefault, stream));
    // ... and then the payload object
    cudaCheck(cudaMemcpyAsync(data.esproductDevice, data.esproduceHost, sizeof(ESProductExampleCUDA), cudaMemcpyDefault, stream));
});

  // Returns the payload object on the memory of the current device
  return data.esproductDevice;
}

// Destructor frees all member pointers
ESProductExampleCUDA::GPUData::~GPUData() {
  if(esproductHost != nullptr) {
    cudaCheck(cudaFree(esproductHost->someData));
    cudaCheck(cudaFreeHost(esproductHost));
  }
  cudaCheck(cudaFree(esProductDevice));
}

```
