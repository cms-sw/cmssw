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

To be written.