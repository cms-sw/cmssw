# PerfTools/MaxMemoryPreload Description

## Introduction

This package generated a library that is meant to be LD_PRELOADed along with libPrefToolsAllocMonitorPreload.so which
uses `cms::perftools::AllocMonitorRegistry` to register a monitor before an application begins. When the application
ends the monitor reports statistics about the allocations and deallocations.

## Usage

To use the package, one must issue the following LD_PRELOAD command before running the application (bash version
shown below)
```
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so libPerfToolsMaxMemoryPreload.so"
```

the order is important.

### Pausing the monitoring

It is possible to temporarily pause the monitoring by instrumenting the target application by definining the following functions
```cpp
// in some header,
void pauseMaxMemoryPreload();
void unpauseMaxMemoryPreload();

// in an implementation source file
void pauseMaxMemoryPreload() {
}
void unpauseMaxMemoryPreload() {
}
```
and then using these in code
```cpp
  ...
  pauseMaxMemoryPreload();
  // code that should be excluded from the monitoring
  unpauseMaxMemoryPreload();
  ...
```

The trick is that by default these functions are defined in the application, and the functions do nothing. The `libPerfToolsMaxMemoryPreload.so` provides also the same functions that actually pause the data collection, and the LD_PRELOADing makes the application to call the functions within `libPerfToolsMaxMemoryPreload.so`.

It is recommended to not pause the monitoring within a multithreaded section, because that could result in unexpected results, because the pausing setting is global.

## Reporting
When the application ends, the monitor will report the following to standard error:

- Total amount of bytes requested by all allocation calls during the job. Note that actual _used_ allocation can be greater than requested as the allocator may require additional memory be assigned.
- The maximum amount of _used_ allocated memory that was in use at one time.
- The amount of _used_ memory allocated during the job that has yet to be reclaimed by calling deallocation.
- Number of calls made to allocation functions.
- Number of calls made to deallocation functions.

This service is multi-thread safe. Note that when run multi-threaded the maximum reported value will vary from job to job.

If a job forks processes, the forked processes will also report the above information.
