# PerfTools/AllocMonitor Description

## Introduction

This package works with the PerfTools/AllocMonitorPreload package to provide a general facility to watch allocations and deallocations.
This is accomplished by using LD_PRELOAD with libPerfToolsAllocMonitorPreload.so and registering a class inheriting from `AllocMonotorBase`
with `AllocMonitorRegistry`. The preloaded library puts in proxies for the C and C++ allocation methods (and forwards the calls to the
original job methods). These proxies communicate with `AllocMonitorRegistry` which, in turn, call methods of the registered monitors.

## Extending

To add a new monitor, one inherits from `cms::perftools::AllocMonitorBase` and overrides the `allocCalled` and
`deallocCalled` methods.

- `AllocMonitorBase::allocCalled(size_t iRequestedSize, size_t iActualSize)` : `iRequestedSize` is the number of bytes being requested by the allocation call. `iActualSize` is the actual number of bytes returned by the allocator. These can be different because of alignment constraints (e.g. asking for 1 byte but all allocations must be aligned on a particular memory boundary) or internal details of the allocator.

- `AllocMonitorBase::deallocCalled(size_t iActualSize)` : `iActualSize` is the actual size returned when the associated allocation was made. NOTE: the glibc extended interface does not provide a way to find the requested size base on the address returned from an allocation, it only provides the actual size.

When implementing `allocCalled` and `deallocCalled` it is perfectly fine to do allocations/deallocations. The facility
guarantees that those internal allocations will not cause any callbacks to be send to any active monitors.


To add a monitor to the facility, one must access the registry by calling the static method
`cms::perftools::AllocMonitorRegistry::instance()` and then call the member function
`T* createAndRegisterMonitor(ARGS&&... iArgs)`. The function will internally create a monitor of type `T` (being careful
to not cause callbacks during the allocation) and pass the arguments `iArgs` to the constructor.

The monitor is owned by the registry and should not be deleted by any other code. If one needs to control the lifetime
of the monitor, one can call `cms::perftools::AllocMonitorRegistry::deregisterMonitor` to have the monitor removed from
the callback list and be deleted (again, without the deallocation causing any callbacks).

## General usage

To use the facility, one needs to use LD_PRELOAD to load in the memory proxies before the application runs, e.g.
```
LD_PRELOAD=libPerfToolsAllocMonitorPreload.so cmsRun some_config_cfg.py
```

Internally, the program needs to register a monitor with the facility. When using `cmsRun` this can most easily be done
by loading a Service which setups a monitor. If one fails to do the LD_PRELOAD, then when the monitor is registered, the
facility will throw an exception.

It is also possible to use LD_PRELOAD to load another library which auto registers a monitor even before the program
begins. See PerfTools/MaxMemoryPreload for an example.

## Services

### SimpleAllocMonitor
This service registers a monitor when the service is created (after python parsing is finished but before any modules
have been loaded into cmsRun) and reports its accumulated information when the service is destroyed (services are the
last plugins to be destroyed by cmsRun). The monitor reports
- Total amount of bytes requested by all allocation calls
- The maximum amount of _used_ (i.e actual size) allocated memory that was in use by the job at one time.
- Number of calls made to allocation functions while the monitor was running.
- Number of calls made to deallocation functions while the monitor was running.
This service is multi-thread safe. Note that when run multi-threaded the maximum reported value will vary from job to job.


### EventProcessingAllocMonitor
This service registers a monitor at the end of beginJob (after all modules have been loaded and setup) and reports its accumulated information at the beginning of endJob (after the event loop has finished but before any cleanup is done). This can be useful in understanding how memory is being used during the event loop. The monitor reports
- Total amount of bytes requested by all allocation calls during the event loop
- The maximum amount of _used_ (i.e. actual size) allocated memory that was in use in the event loop at one time.
- The amount of _used_ memory allocated during the loop that has yet to be reclaimed by calling deallocation.
- Number of calls made to allocation functions during the event loop.
- Number of calls made to deallocation functions during the event loop.
This service is multi-thread safe. Note that when run multi-threaded the maximum reported value will vary from job to job.

### HistogrammingAllocMonitor
This service registers a monitor when the service is created (after python parsing is finished but before any modules
have been loaded into cmsRun) and reports its accumulated information when the service is destroyed (services are the
last plugins to be destroyed by cmsRun). The monitor histograms the values into bins of number of bytes where each
bin is a power of 2 larger than the previous. The histograms made are
- Amount of bytes requested by all allocation calls
- Amount of bytes actually used by all allocation calls
- Amount of bytes actually returned by all deallocation calls
This service is multi-thread safe. Note that when run multi-threaded the maximum reported value will vary from job to job.
