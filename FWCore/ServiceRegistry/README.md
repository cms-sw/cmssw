# ServiceRegistry

This page documents the [`ActivityRegistry`](interface/ActivityRegistry.h) signals.

Signal actions are functions registered to be called when a signal is emitted. When a signal is emitted, all signal actions are called.

**Generally it is best for the signal actions to not throw exceptions.** Framework will catch and handle such exceptions (by gracefully terminating the process), but the detailed behavior is unspecified and should not be relied upon. If any of the actions throws an exception, the `emit()` call throws the first of those exceptions.

For the signal types that have both `Pre` and `Post` signals, the `Pre` signals are emitted before the corresponding operation (e.g. constructing modules) and the `Post` signals are emitted after. If all `Pre` signal actions succeed and the operation itself succeeds, the corresponding `Post` signal actions are guaranteed to be called. If the operation itself throws an exception, it is desired that the corresponding `Post` signal actions are called, but this is not guaranteed and should not be relied upon. If any of the `Pre` signal actions throws an exception, the operation itself is not called (guaranteed), and it is desired that the corresponding `Post` signal actions are called (not guaranteed, do not rely). From the point of view of a Service that registers signal actions, it is desired that if a `Pre` signal action is called, the corresponding `Post` signal action is also called, but this is not guaranteed and should not be relied upon.

**The `Post` signal actions should not throw exceptions.** In many cases those would lead to `std::terminate()` being called.

See also the [SWGuideExceptionsInBeginEndTransitions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideExceptionsInBeginEndTransitions) Twiki page for more information on the exception handling between Begin and End transitions.

## Signal Types

The naming of `ActivityRegistry` member objects, that are used for emitting the signals, follow a pattern where the first character is in lower case, and `Signal_` is appended to the end of the signal type name.

The order of the signals in this list is an approximate order in which they would be emitted in a typical single-thread job. The order should be kept consistent with the `ActivityRegistry.h`. 



### Construction phase

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `PostServicesConstruction` | - | Emitted after all services have been constructed. |
| `{Pre,Post}EventSetupModulesConstruction` | Yes | Emitted before and after constructing EventSetup modules. |
| `{Pre,Post}ESModuleConstruction` | Yes | Emitted before and after constructing an EventSetup module (ESProducer or ESSource). |
| `PostESModuleRegistration` | - | Emitted after an ESModule provider has been registered with the EventSetupProvider. |
| `{Pre,Post}ModulesAndSourceConstruction` | Yes | Emitted before and after concurrent construction of the Source and the EDModules. |
| `{Pre,Post}SourceConstruction` | Yes | Emitted before and after construction of the Source. |
| `{Pre,Post}OpenFile` | Yes | `PoolSource`, `EmbeddedRootSource`, `RNTupleTempSource`, and `EmbeddedRNTupleTempSource` emit these signals  before and after opening a new input file (either primary or secondary). Note that the first signal may be emitted during the Source construction, and the later signals after the `BeginProcessing`. |
| `{Pre,Post}ModuleConstruction` | Yes | Emitted before and after constructing an EDModule. |
| `{Pre,Post}FinishSchedule` | Yes | Emitted before and after finalizing the Schedule. |
| `{Pre,Post}PrincipalsCreation` | Yes | Emitted before and after creating the Principals. |

### Startup phase

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `Preallocate` | - | Emitted with system bounds information for resource preallocation. |
| `{Pre,Post}ScheduleConsistencyCheck` | Yes | Emitted before and after checking module dependency correctness and deleting non-consumed unscheduled EDModules. |
| `{Pre,Post}ModuleDestruction` | Yes | Emitted before and after an EDModule is destructed because there was no data or control flow dependence on it. |
| `{Pre,Post}EventSetupConfigurationFinalized` | Yes | Emitted before and after finishing the EventSetup configuration. |
| `EventSetupConfiguration` | - | Emitted with EventSetup configuration details. |
| `{Pre,Post}ModulesInitializationFinalized` | Yes | Emitted before and after finalizing module initialization. |
| `{Pre,Post}BeginJob` | Yes | Emitted before and after running `beginJob()` for all modules. |
| `{Pre,Post}ModuleBeginJob` | Yes | Emitted before and after each module's `beginJob()` method is called. |
| `LookupInitializationComplete` | - | Emitted with module data dependency details. |
| `{Pre,Post}BeginStream` | Yes | Emitted before and after running `beginStream()` for all modules. |
| `{Pre,Post}ModuleBeginStream` | Yes | Emitted before and after each module's `beginStream()` method is called on a specific stream. |

### Data processing phase

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `BeginProcessing` | - | Emitted right before the data processing phase is started. |
| `{Pre,Post}SourceNextTransition` | Yes | Emitted before and after the input source determines the next transition type (event, run, lumi, etc.). |
| `{Pre,Post}OpenOutputFiles` | Yes | Emitted before and after opening OutputModules' output files. |
| `{Pre,Post}BeginProcessBlock` | Yes | Emitted before and after a global `BeginProcessBlock` transition for all modules. |
| `{Pre,Post}ModuleBeginProcessBlock` | Yes | Emitted before and after each EDModule's global `beginProcessBlock()` method is called. |
| `{Pre,Post}SourceProcessBlock` | Yes | Emitted before and after the `InputSource` reads a `ProcessBlock`. |
| `{Pre,Post}AccessInputProcessBlock` | Yes | Emitted before and after a global `AccessInputProcessBlock` transition for all modules. |
| `{Pre,Post}ModuleAccessInputProcessBlock` | Yes | Emitted before and after each EDModule's `accessInputProcessBlock()` method is called to read data from input ProcessBlocks. |
| `ESSyncIOVQueuing` | - | After #51465 the signal should be removed. |
| `{Pre,Post}ESSyncIOV` | Yes | Emitted before and after finding the proper EventSetup IOVs from the ESSources. |
| `{Pre,Post}SourceRun` | Yes | Emitted before and after the `InputSource` reads a `Run`. |
| `{Pre,Post}GlobalBeginRun` | No | Emitted before and after running the global `BeginRun` transition for all modules. |
| `{Pre,Post}ModuleGlobalPrefetching` | No | Emitted before and after asynchronously prefetching data products needed by each EDModule for global {BeginRun, EndRun, BeginLumi, EndLumi} transitions . |
| `{Pre,Post}ESModulePrefetching` | No | Emitted before and after prefetching data dependencies for an ESModule. |
| `{Pre,Post}ESModuleAcquire` | Yes | Emitted before and after an ESModule's "acquire" method for external work. |
| `{Pre,Post}ESModule` | Yes | Emitted before and after running an ESModule's "produce" or `setIntervalFor()` method. |
| `{Pre,Post}ModuleGlobalBeginRun` | Yes | Emitted before and after each EDModule's global `beginRun()` method is called. |
| `{Pre,Post}StreamBeginRun` | No | Emitted before and after running the stream `BeginRun` transition for all modules on a specific stream. |
| `{Pre,Post}ModuleStreamPrefetching` | No | Emitted before and after asynchronously prefetching data products needed by each EDModule for stream {BeginRun, EndRun, BeginLumi, EndLumi} transitions. |
| `{Pre,Post}ModuleStreamBeginRun` | Yes | Emitted before and after each EDModule's stream `beginRun()` method is called on a specific stream. |
| `{Pre,Post}SourceLumi` | Yes | Emitted before and after the `InputSource` reads a `LuminosityBlock`. |
| `{Pre,Post}GlobalBeginLumi` | No | Emitted before and after running the global `BeginLuminosityBlock` transition for all modules. |
| `{Pre,Post}ModuleGlobalBeginLumi` | Yes | Emitted before and after each EDModule's global `beginLuminosityBlock()` method is called on a specific stream. |
| `{Pre,Post}StreamBeginLumi` | No | Emitted before and after running the stream `BeginLuminosityBlock` transition for all modules on a specific stream. |
| `{Pre,Post}ModuleStreamBeginLumi` | Yes | Emitted before and after each EDModule's stream `beginLuminosityBlock()` method is called on a specific stream. |
|||
| `{Pre,Post}SourceEvent` | Yes | Emitted before and after the `InputSource` reads an `Event` (effectively the metadata). The `PoolSource` and `RNTupleTempSource` emit these signals also when they read an `Event` for the secondary input. |
| `{Pre,Post}Event` | No | Emitted before and after running the `Event` transition for all modules on a specific stream. |
| `{Pre,Post}EventReadFromSource` | Yes | `PoolSource`, `EmbeddedRootSource`,  `RNTupleTempSource`, and `EmbeddedRNTupleTempSource` emit these signals before and after reading event provenance and event data products via the delayed reading. In case of prompt reading, the signals are emitted when the data products are retrieved from the cache of the prompt reader. `RepeatingCachedRootSource` emits these signal when retrieving event data products from its cache. |
| `{Pre,Post}PathEvent` | No | Emitted before and after processing a Path for an Event. |
| `{Pre,Post}ModuleEventPrefetching` | No | Emitted before and after each module's prefetching of data products needed by the EDModule for an Event. |
| `{Pre,Post}ModuleEventAcquire` | Yes | Emitted before and after each ExternalWork-using EDModule's `acquire()` method. |
| `{Pre,Post}ModuleEventDelayedGet` | Yes | Emitted before and after an EDModule performs a delayed get operation to retrieve a data product. This can only | `{Pre,Post}ModuleEvent` | Yes | Emitted before and after running each EDModule's Event processing method (e.g., `produce()`, `filter()`, `analyze()`). |
| `{Pre,Post}ModuleTransformPrefetching` | No | Emitted before and after each Transformer-using EDModule's prefetching of data products needed by the transformer. |
| `{Pre,Post}ModuleTransformAcquiring` | Yes | Emitted before and after each Transformer-using EDModules' `transformAsync`'s "acquire" function. |
| `{Pre,Post}ModuleTransform` | Yes | Emitted before and after each Transformer-using EDModules's `transform` function. |
occur when accessing data from `edm::Ref`-style objects. |
| `{Pre,Post}ClearEvent` | Yes | Emitted before and after deleting the Event data products after the Event has been processed. Note that data products deleted early are not signalled. |
|||
| `{Pre,Post}StreamEndLumi` | No | Emitted before and after running the stream `EndLuminosityBlock` transition for all modules on a specific stream. |
| `{Pre,Post}ModuleStreamEndLumi` | Yes | Emitted before and after each EDModule's stream `endLuminosityBlock()` method is called on a specific stream. |
| `{Pre,Post}GlobalEndLumi` | No | Emitted before and after running the global `EndLuminosityBlock` transition for all modules. |
| `{Pre,Post}ModuleGlobalEndLumi` | Yes | Emitted before and after each EDModule's global `endLuminosityBlock()` method is called.  |
| `{Pre,Post}GlobalWriteLumi` | No | Emitted before and after the OutputModules write the data of a LuminosityBock. |
| `{Pre,Post}ModuleWriteLumi` | Yes | Emitted before and after each OutputModule's `doWriteLuminosityBlock()` method, that writes LuminosityBlock data to output. |
| `{Pre,Post}StreamEndRun` | No | Emitted before and after running the stream `EndRun`  transition for all modules on a specific stream.  |
| `{Pre,Post}ModuleStreamEndRun` | Yes | Emitted before and after each EDModule's stream `endRun()` method is called on a specific stream. |
| `{Pre,Post}GlobalEndRun` | No | Emitted before and after runninge the global `EndRun` transition for all modules. |
| `{Pre,Post}ModuleGlobalEndRun` | Yes | Emitted before and after each EDModule's global `endRun()` method is called. |
| `{Pre,Post}GlobalWriteRun` | No | Emitted before and after the OutputModules write the data of a Run. |
| `{Pre,Post}ModuleWriteRun` | Yes | Emitted before and after each OutputModule's `doWriteRun()` method, that writes Run data to output. |
| `{Pre,Post}WriteProcessBlock` | No | Emitted before and after OutputModules write the data of a ProcessBlock. |
| `{Pre,Post}ModuleWriteProcessBlock` | Yes | Emitted before and after each OutputModule's `doWriteProcessBlock()` method, that writes ProcessBlock data to output. |
| `{Pre,Post}CloseFile` | Yes | `PoolSource` and `RNTupleTempSource`  emit these signals before and after closing a primary input file file. |
| `{Pre,Post}EndProcessBlock` | Yes | Emitted before and after a global `EndProcessBlock` transition for all modules. |
| `{Pre,Post}ModuleEndProcessBlock` | Yes | Emitted before and after each EDModule's global `endProcessBlock()` method is called. |
| `{Pre,Post}CloseOutputFiles` | Yes | Emitted before and after closing OutputModules' output files. |
| `EndProcessing` | - | Emitted right after the data processing phase is finished. |

### Shutdown phase

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `{Pre,Post}EndStream` | Yes | Emitted before and after running `endStream()` for all modules. |
| `{Pre,Post}ModuleEndStream` | Yes | Emitted before and after each module's `endStream()` method is called on a specific stream. |
| `{Pre,Post}EndJob` | Yes | Emitted before and after running `endJob()` for all modules. |
| `{Pre,Post}ModuleEndJob` | Yes | Emitted before and after each module's `endJob()` method is called. |

### Early termination signals

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `PreStreamEarlyTermination` | - | Emitted when an exception is thrown from any stream transition. |
| `PreGlobalEarlyTermination` | - | Emitted when an exception is thrown from any global transition. |
| `PreSourceEarlyTermination` | - | Emitted when an external termination request is received. |
| `JobFailure` | - | To be removed (only called from `FWCore/Services/test/servicesJobReport_t.cpp`) |
