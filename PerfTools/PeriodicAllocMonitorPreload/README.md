# PerfTools/PeriodicAllocMonitorPreload Description

## Introduction

This package generated a library that is meant to be LD_PRELOADed along with libPrefToolsAllocMonitorPreload.so which
uses `cms::perftools::AllocMonitorRegistry` to register a monitor before an application begins. As the application
is running this library will periodically report statistics about the allocations and deallocations to a file.

## Usage

To use the package, one must issue the following LD_PRELOAD command before running the application (bash version
shown below)
```
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so libPerfToolsPeriodicAllocMonitorPreload.so"
```

the order is important.

Both the base part of the file name to be used as well as the interval between samples can be set via environment variables

| *Variable* | *Description* |
| --- | --- |
| PAM_FILENAME | The base part of the file name to write. The process id of the job will be appended to the base name followed by `.csv`. The reason for the process name is to handle the case where the function forks another process. In that case we need the new process to write to its own file. If the environment variable is not set, the value of  "periodic_alloc_" will be used. |
| PAM_INTERVAL_MS | this is the number of milliseconds the system should wait before reporting the information. If the environment variable is not set, the value of "1000" will be used. |

## Reporting
The output file contains the following information on each line
- The time, in milliseconds, since the service was created
- Total amount of bytes requested by all allocation calls since the service started
- The maximum amount of _used_ (i.e. actual size) allocated memory that has been seen up to this point in the job
- The amount of _used_ memory allocated at the time the report was made.
- The largest single allocation request that has been seen up to the time of the report
- Number of calls made to allocation functions
- Number of calls made to deallocation functions

If a job forks processes, the forked processes will also report the above information.
