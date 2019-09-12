# Prototype for CMSSW interface to heterogenous algorithms

## Introduction

This package contains a prtotype for the CMSSW interface to
heterogeneous algorithms. The current implementation is, in a sense, a
mini-framework between the CMSSW core framework and the heterogeneous
algorithms.

More details can be found from the sub-package specific README files (when they get added).

## Sub-packages

* [`CUDACore`](../CUDACore) CUDA-specific core components
  - *TODO:* Do we actually need this separate from `CUDAServices`? Which one to keep?
* [`CUDAServices`](../CUDAServices) Various edm::Services related to CUDA
* [`CUDAUtilities`](../CUDAUtilities) Various utilities for CUDA kernel code
* [`Producer`](#heterogeneousedproducer) Core of the mini-framework for code organization: a base EDProducer class with algorithm scheduling to devices
* [`Product`](../Product) Core of the mini-framework for data products

## Design goals

1. Same module configuration should work on all machines (whether they have heterogeneous devices or not)
2. Keep track of where data is located
3. Run algorithms on the device where their input is located if possible 
4. Transfer temporary/transient data to CPU only if needed
5. Abstract away repeated boilerplate code

## Design considerations

All below is assuming we do not touch the core CMSSW framework (that
is left for a later exercise when we know better what exactly we want
to do).

1. The device-specific algorithms must be implemented and scheduled to the device(s) within a single EDProducer
2. Need a special product keeping track of the data location
3. Information of all the input heterogeneous devices must be propagated to the point deciding the device to run the algorithm
4. The special product e.g. holds functions to do the transfer that are called if/when needed

## General TODO items

There are also many, more specific TODO items mentioned in comments
within the code. The items below are more general topics (in no
particular order).

* Improve algorithm-to-device scheduling
  - Currently if an algoritm has a GPU implementation and system has a
    GPU, the algorithm is always scheduled to the GPU
    * This may lead to under-utilization of the CPU if most of the
      computations are offloaded to the GPU
  - An essential question for making this scheduling more dynamic is
    what exactly (we want) it (to) means that a "GPU is too busy" so
    it is better to run the algorithm on a CPU
  - Possible ideas to explore
    * Check the CUDA device utilization (see also monitoring point below)
      - Past/current state does not guarantee much about the near future
    * Use "tokens" as a resource allocation mechanism
      - How many tokens per device?
      - What if there no free tokens now but one becomes available after 1 ms?
    * In acquire, if GPU is "busy", put the EDProducer to a queue of
      heterogeneous tasks. When GPU "becomes available", pick an
      EDProducer from the queue and run it in GPU. If CPU runs out of
      job, pick an EDProducer from the queue and run it in CPU.
      - How to define "busy" and "becomes available"?
      - How to become aware that CPU runs out of job?
        * Can we create a TBB task that is executed only if there is nothing else to do?
      - How does this interact with possible other devices? E.g. if an algorithm has implementations for CPU, GPU, and FPGA?
* Improve edm::Stream-to-CUDA-device scheduling
  - Currently each edm::Stream is assigned "statically" to each CUDA device in a round-robin fastion
    * There is no load balancing so throughput and utilization will not be optimal
  - The reasons for bothering with this is that the `cudaMalloc` is a
    heavy operation (see next point) not to be called for each event.
    Instead we preallocate the buffers the algorithms need at the
    initialization time. In the presence of multiple devices this
    pre-allocation leads to a need to somehow match the edm::Streams
    and devices.
  - Possible ideas to explore
    * Naively we could allocate a buffer per edm::Stream in each CUDA device
      - Amount of allocated memory is rather excessive
    * For N edm::Streams and M GPUs, allocate int((N+1)/M) buffers on each device, eventually pick the least utilized GPU
      - No unnecessary buffers
      - Need a global list/queue of these buffers per module
        * Can the list be abstracted? If not, this solution scales poorly with modules
    * Our own CUDA memory allocator that provides a fast way to allocate scratch buffers 
      - Allows allocating the buffers on-demand on the "best-suited" device
* Our own CUDA memory allocator
  - A `cudaMalloc` is a global synchronization point and takes time,
    so we want to minimize their calls. This is the main reason to
    assign edm::Streams to CUDA devices (see previous point).
  - Well-performing allocators are typically highly non-trivial to construct
* Conditions data on GPU
  - Currently each module takes care of formatting, transferring, and updating the conditions data to GPU
  - This is probably good-enough for the current prototyping phase, but what about longer term?
    * How to deal with multiple devices, multiple edm::Streams, and multiple lumi sections in flight?
    * Do we need to make EventSetup aware of the devices? How much do the details depend on device type?
* Add possibility to initiate the GPU->CPU transfer before the CPU product is needed
  - This would enable overlapping the GPU->CPU transfer while CPU is busy
    with other work, so the CPU product requestor would not have to wait
* Improve configurability
  - E.g. for preferred device order?
* Add fault tolerance
  - E.g. in a case of a GPU running out of memory continue with CPU
  - Should be configurable
* Add support for multiple heterogeneous inputs for a module
  - Currently the device scheduling is based only on the "first input"
  - Clearly this is inadequate in general and needs to be improved
  - Any better suggestions than taking `and` of all locations?
* Improve resource monitoring
  - E.g. for CUDA device utilization
    * https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g540824faa6cef45500e0d1dc2f50b321
    * https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
  - Include in `CUDAService` or add a separete monitoring service?
* Add support for a mode similar to `tbb::streaming_node`
  - Current way of `HeterogeneousEDProducer` with the `ExternalWork` resembles `tbb::async_node`
  - In principle the `streaming_node`-way would, for a chain of
    GPU-enabled modules, allow the GPU to immediately continue to the
    next module without waiting the code path to go through the CMSSW
    framework
* Add support for more devices
  - E.g. OpenCL, FPGA, remote offload
* Explore the implementation of these features into the core CMSSW framework
  - E.g. HeterogeneousProduct would likely go to edm::Wrapper
* Explore how to make core framework/TBB scheduling aware of heterogenous devices

# HeterogeneousEDProducer

`HeterogeneousEDProducer` is implemented as a `stream::EDProducer` using the
[`ExternalWork` extension](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#edm_ExternalWork).

## Configuration

`HeterogeneousEDProducer` requires `heterogeneousEnabled_` `cms.PSet`
to be available in the module configuration. The `PSet` can be used to
override module-by-module which devices can be used by that module.
The structure of the `PSet` are shown in the following example

```python
process.foo = cms.EDModule("FooProducer",
    a = cms.int32(1),
    b = cms.VPSet(...),
    ...
    heterogeneousEnabled_ = cms.untracked.PSet(
        GPUCuda = cms.untracked.bool(True),
        FPGA = cms.untracked.bool(False),       # forbid the module from running on a device type (note that we don't support FPGA devices at the moment though)
        force = cms.untracked.string("GPUCuda") # force the module to run on a specific device type
    )
)
```

The difference between the boolean flags and the `force` parameter is the following
* The boolean flags control whether the algorithm can be scheduled on the individual device type or not
* The `force` parameter implies that the algorithm is always scheduled on that device type no matter what. If the device type is not available on the machine, an exception is thrown.

Currently, with only CUDA GPU and CPU support, this level of configurability is a bit overkill though.

## Class declaration

In order to use the `HeterogeneousEDProducer` the `EDProducer` class
must inherit from `HeterogeneousEDProducer<...>`. The devices, which
the `EDProducer` is supposed to support, are given as a template
argument via `heterogeneous::HeterogeneousDevices<...>`. The usual
[stream producer extensions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#Template_Arguments)
can be also passed via additional template arguments.

```cpp
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h" // needed for heterogeneous::GPUCuda

class FooProducer: public HeterogeneousEDProducer<
  heterogeneous::HeterogeneousDevices<
    heterogeneous::GPUCuda,
    heterogeneous::CPU
  >
  // here you can pass any stream producer extensions
> {
  ...

```

In this example the `FooProducer` declares that prodives
implementations for CUDA GPU and CPU. Note that currently CPU is
mandatory, and it has to be the last argument. The order of the
devices dictates the order that the algorithm is scheduled to the
devices. E.g. in this example, the system runs the algorithm in GPU if
it can, and only if it can not, in CPU. For the list of supported
device types, see the [list below](#devices).

## Constructor

`HeterogeneousEDProducer` needs access to the configuration, so it has
to be passed to its constructor as in the following example

```cpp
FooProducer::FooProducer(edm::ParameterSet const& iConfig):
  HeterogeneousEDProducer(iConfig),
  ...
```
### Consumes

If the `EDProducer` reads any `HeterogeneousProduct`'s
([see more details](#heterogeneousproduct)), the `consumes()` call
should be made along the following

```cpp
class FooProducer ... {
  ...
  EDGetTokenT<HeterogeneousProduct> token_;
};
...
FooProducer::FooProducer(edm::ParameterSet const& iConfig):
  ...
  token_(consumesHeterogeneous(iConfig.getParameter<edm::InputTag>("..."))),
  ...
```

so that `HeterogeneousEDProducer` can inspect the location of input
heterogeneous products to decide on which device to run the algorithm
([see more details](#device-scheduling)).

### Produces

If the `EDProducer` produces any `HeterogeneousProduct`'s
([see more details](#heterogeneousproduct)), the `produces()` call
should be made along the following (i.e. as usual)

```cpp
FooProducer::FooProducer(edm::ParameterSet const& iConfig) ... {
  ...
  produces<HeterogeneousEvent>();
}
```

## fillDescriptions()

`HeterogeneousEDProducer` provides a `fillPSetDescription()` function
that can be called from the concrete `EDProducer`'s
`fillDescriptions()` as in the following example

```cpp
void FooProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // fill desc for other parameters

  HeterogeneousEDProducer::fillPSetDescription(desc);

  descriptions.add("fooProducer", desc);
}
```

## Device scheduling

The per-event scheduling of algorithms is currently as follows
1. If there are no `HeterogeneousProduct` inputs ([see more details](#heterogeneousproduct))
   * Loop over the device types in the order specified in `heterogeneous::HeterogeneousDevices<...>` template arguments
   * Run the algorithm on the first device that is enabled for that module (instance)
2. If there are `HeterogeneousProduct` inputs
   * Run the algorithm on **the device where the data of the first input resides**

## HeterogeneousProduct

The `HeterogeneousProduct` is a transient edm product with the following properties
* placeholder for products (of arbitrary types) in all device types
* tracks the location of the data
* automatic, on-demand transfers from device to CPU 
  - developer has to provide a function to do the transfer and possible data reorganiazation

Some of the complexity exists to avoid ROOT dictionary generation of the concrete product types.

## HeterogeneousEvent

The `HeterogeneousEvent` is a wrapper on top of `edm::Event` to hide
most of the complexity of `HeterogeneousProduct` to make its use look
almost like standard products with `edm::Event`. Some part of the
`edm::Event` interface is implemented (and delegated back to
`edm::Event`) in order to get/put standard products as well.

Here is a short example how to deal use `HeterogeneousProduct` with
`HeterogeneousEvent` (using the same `FooProducer` example as before)

```cpp
class FooProducer ... {
  ...
  // in principle these definitions should be treated like DataFormats
  struct CPUProduct {
    std::vector<int> foo;
  };
  struct GPUProduct {
    float *foo_d; // pointer to GPU memory
  }

  using InputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                             heterogeneous::GPUProduct<GPUProduct>>;
  using OutputType = InputType; // using same input and output only because of laziness
  
  void transferGPUtoCPU(GPUProduct const& gpu, CPUProduct& cpu) const;
};

void FooProducer::produceCPU(edm::HeterogeneousEvent const& iEvent, ...) {
  edm::Handle<CPUProduct> hinput;
  iEvent.getByToken<InputType>(token_, hinput); // note the InputType template argument

  // do whatever you want with hinput->foo;

  auto output = std::make_unique<CPUProduct>(...);
  iEvent.put<OutputType>(std::move(output));    // note the OutputType template argument
}

void FooProducer::acquireGPUCuda(edm::HeterogeneousEvent const& iEvent, ...) {
  edm::Handle<GPUProduct> hinput;
  iEvent.getByToken<InputType>(token_, hinput); // note the InputType template argument

  // do whatever you want with hinput->foo_d;

  auto output = std::make_unique<GPUProduct>(...);
  // For non-CPU products, a GPU->CPU transfer function must be provided
  // In this example it is prodided as a lambda calling a member function, but this is not required
  // The function can be anything assignable to std::function<void(GPUProduc const&, CPUProduct)>
  iEvent.put<OutputType>(std::move(output), [this](GPUProduct const& gpu, CPUProduct& cpu) { // note the OutputType template argument
    this->transferGPUtoCPU(gpu, cpu);
  });
  // It is also possible to disable the GPU->CPU transfer
  // If the data resides on a GPU, and the corresponding CPU product is requested, an exception is thrown
  //iEvent.put<OutputType>(std::move(output), heterogeneous::DisableTransfer); // note the OutputType template argument
}

```


## Devices

This section documents which functions the `EDProducer` can/has to
implement for various devices.

### CPU

A CPU implementation is declared by giving `heterogeneous::CPU` as a
template argument to `heterogeneous::HeterogeneousDevices`. Currently
it is a mandatory argument, and has to be the last one (i.e. there
must always be a CPU implementation, which is used as the last resort
if there are no other devices).

#### Optional functions

There is one optional function

```cpp
void beginStreamCPU(edm::StreamID id);
```

which is called at the beginning of an `edm::Stream`. Usually there is
no need to implement it, but the possibility is provided in case it is
needed for something (as the `stream::EDProducer::beginStream()` is
overridden by `HeterogeneousEDProducer`).

#### Mandatory functions

There is one mandatory function

```cpp
void produceCPU(edm::HeterogeneousEvent& iEvent, edm::EventSetup const& iSetup);
```

which is almost equal to the usual `stream::EDProducer::produce()`
function. It is called from `HeterogeneousEDProducer::produce()` if
the device scheduler decides that the algorithm should be run on CPU
([see more details](#device-scheduling)). The first argument is
`edm::HeterogeneousEvent` instead of the usual `edm::Event`
([see more details](#heterogeneousevent)).

The function should read its input, run the algorithm, and put the
output to the event.

### CUDA GPU

A CUDA GPU implementation is declared by giving
`heterogeneous::GPUCuda` as a template argument to
`heterogeneous::HeterogeneousDevices`. The following `#include` is
also needed
```cpp
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
```
#### Optional functions

There is one optional function

```cpp 
void beginStreamGPUCuda(edm::StreamID id, cuda::stream_t<>& cudaStream);
```

which is called at the beginning of an `edm::Stream`. **If the
algorithm has to allocate memory buffers for the duration of the whole
job, the recommended place is here.** The current CUDA device is set
by the framework before the call, and all asynchronous tasks should be
enqueued to the CUDA stream given as an argument.

Currently the very same CUDA stream object will be given to the
`acquireGPUCuda()` and `produceGPUCuda()` for this `edm::Stream`. This
may change in the future though, in which case it will still be
guaranteed that the CUDA stream here will be synchronized before the
first call to `acquireGPUCuda()`.

#### Mandatory functions

There are two mandatory functions:

```cpp
void acquireGPUCuda(edm::HeterogeneousEvent const& iEvent, edm::EventSetup const& iSetup, cuda::stream_t<>& cudaStream);
```

The `acquireGPUCuda()` is called from
`HeterogeneousEDProducer::acquire()` if the device scheduler devices
that the algorithm should be run on a CUDA GPU
([see more details](#device-scheduling)). The function should read
the necessary input (which may possibly be already on a GPU,
([see more details](#heterogeneousproduct)), and enqueue the
*asynchronous* work on the CUDA stream given as an argument. The
current CUDA deviceis set by the framework before the call. After the
`acquireGPUCuda()` returns, framework will itself enqueue a callback
function to the CUDA stream that will call
`edm::WaitingTaskWithArenaHolder::doneWaiting()` to signal to the
framework that this `EDProducer` is ready to transition to
`produce()`.

Currently the very same CUDA stream will be given to the
`produceGPUCuda()`.


```cpp
void produceGPUCuda(edm::HeterogeneousEvent& iEvent, edm::EventSetup const& iSetup, cuda::stream_t<>& cudaStream);
```

The `produceGPUCuda()` is called from
`HeterogeneousEDProducer::produce()` if the algorithm was run on a
CUDA GPU. The function should do any necessary GPU->CPU transfers,
post-processing, and put the products to the event (for passing "GPU
products" [see here](#heterogeneousproduct)).

#### Memory allocations

The `cudaMalloc()` is somewhat heavy function (synchronizing the whole
device, among others). The current strategy (although not enforced by
the framework) is to allocate memory buffers at the beginning of a
job. It is recommended to do these allocations in the
`beginStreamGPUCuda()`, as it is called exactly once per job per
`stream::EDProducer` instance, and it is the earliest point in the
framework where we have the concept of `edm::Stream` so that the
framework can assign the `edm::Stream`s to CUDA devices
([see more details](#multipledevices)).

Freeing the GPU memory can be done in the destructor as it does not
require any special support from the framework.

#### Multiple devices

Currently `edm::Stream`'s are statically assigned to CUDA devices in a
round-robin fashion. The assignment is done at the `beginStream()`
time before calling the `EDProducer` `beginStreamDevice()` functions.

Technically "assigning `edm::Stream`" means that each relevant
`EDProducer` instance of that `edm::Stream` will hold a device id of
`streamId % numberOfDevices`.

### Mock GPU

The `GPUMock` is intended only for testing of the framework as a
something requiring a GPU-like interface but still ran on the CPU. The
documentation is left to be the code itself.
