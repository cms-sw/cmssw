# PerfTools/ThresholdAbortAllocMonitorPreload Description

## Introduction

This package generated a library that is meant to be LD_PRELOADed along with libPrefToolsAllocMonitorPreload.so which
uses `cms::perftools::AllocMonitorRegistry` to register a monitor before an application begins. When the application
request an allocation that falls within the threshold range, the process will either abort or a special named function
(`break_threshold_abort_alloc_monitor`) will be called.

## Usage

To use the package, one must issue the following LD_PRELOAD command before running the application (bash version
shown below)
```
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so libPerfToolsThresholdAbortAllocMonitorPreload.so"
```

the order is important.

The threshold range, how many triggered allocations should be skipped  and whether to abort or call `break_threshold_abort_alloc_monitor` is set via environment variables.

| *Variable* | *Description* |
| --- | --- |
| TAAM_MIN | The minimum size (in bytes) of an allocation request which will trigger the system. |
| TAAM_MAX | The maximum size (in bytes) of an allocation request which will trigger the system. A value of 0 means no max is used for the trigger (just the min). Default is 0. |
| TAAM_SKIP | Number of triggers (i.e. allocations that fell within the threshold range) to skip before calling abort or `break_threshold_abort_alloc_monitor`. Default is 0. |
| TAAM_BREAK | If set, will call `break_threshold_abort_alloc_monitor` rather than abort when reach the beyond the skipped triggers. |

If a job forks processes, the forked processes will also be subjected to the triggering.
