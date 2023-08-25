# PerfTools/AllocMonitorPreload Description

## Introduction

This package works with the PerfTools/AllocMonitor package to provide a general facility to watch allocations and deallocations. See the README.md in that package for details on how to use the facility.

## Technical Details

This package overrides the standard C and C++ allocation and deallocation functions. These overridden functions call the
appropriate methods of `cms::perftools::AllocMonitorRegistry` to allow monitoring of memory allocations and
deallocations. The overridden C and C++ standard methods use `dlsym` to find the original functions (i.e. what ever
versions of those methods were compiled into the executable) and then call those original functions to do the actual
allocation/deallocation.

To support standard library C++ on linux, one only needs to override the standard C methods to intercept all
allocations/deallocations. However, to intercept calls for jemalloc or tcmalloc, one must also override the C++
methods. This is complicated as one must call `dlsym` using the _mangled_ names of the C++ methods. As the exact
mangled name can be different on different operating systems or CPU types the exact name used will have to be updated
in this code to use different systems. `AllocMonitorRegistry` makes sure that if a standard function calls another
standard function to do the actual work, that only one callback will be issued.

There is no C or C++ standard method one can call to ask how much actual memory is associated with a given address
returned by an allocator. To provide such information, we use the GNU standard `malloc_usable_size` method. To have
this facility support additional operating systems, an equivalent method would be needed.

The facility starts and stops calls to the `cms::perftools::AllocMonitorRegistry` via the use of the functions `alloc_monitor_start` and `alloc_monitor_stop`. The `AllocMonitorRegistry` use `dlsym` to locate these methods (avoiding link
time dependencies with this package) and if the method is not available by the first request to register a monitor the
code throws an exception. The destructor of `AllocMonitorRegistry` calls the stop method. In this way, the facility can
never call methods on `AllocMonitorRegistry` when the registry is not available.

