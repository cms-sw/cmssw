# PerfTools/VTune Description

## Introduction

This package provides services that can be used in conjunction with VTune profiler.

## Services

### VTuneAnnotateService:

This service annotates module event execution with ITT tasks. The label of the task is constructed from the module name and module label. It uses itt_task_begin/itt_task_end at the start/end of module event execution. The ITT tasks are created and destroyed automatically at the start and end of each thread so cmsRun does not need to be run single threaded. In the VTune UI bottom-up view, these tasks will appear under the ITT domain specified by the ittDomain parameter.

#### parameters
    moduleNames: List of module labels to annotate, defaults to empty
    allModules: Annotate all modules, defaults to false
    ittDomain: ITT domain for the tasks, defaults to CMSSW.ModuleTracker

If `allModules=true` the list of module names is ignored.

If `allModules=false` and the list of module names is empty no modules are annotated.

### VTuneFilterService

This service enables VTune profiling for selected modules or all modules using the allModules option. It uses itt_resume/itt_pause at the start/end of the module event execution. VTune must be started paused. To profile only program execution module event processing, cmsRun must be run single threaded as itt_resume resums profiling and itt_pause pauses profiling for the whole process. Although profiling is paused VTune is still responding to SIGPROF and generating a stack trace. The stack trace generation will cause a segfault in tracing JITed code, e.g. from TensorFlow or Eigen.

#### parameters
    moduleNames: List of module labels or module class names to enable profiling for, defaults to empty
    allModules: Enable profiling for all modules. If set to false no modules will be profiled unless moduleNames contains valid module labels or module class names.

If `allModules=true` the list of module names is ignored.

If `allModules=false` and the list of module names is empty no module is profiled.

