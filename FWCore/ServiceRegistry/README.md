# ServiceRegistry

This page documents the [`ActivityRegistry`](interface/ActivityRegistry.h) signals.

Generally signal actions (functions registered to watch the signals) should not throw exceptions. When a signal is emitted, all signal actions are called. If at least one of the actions throws an exception, the first exception thrown is propagated through the signal `emit()` call.

For the signal types that have both `Pre` and `Post` signals, the `Pre` signals are emitted before the corresponding operation (e.g., constructing modules) and the `Post` signals are emitted after. Generally if a `Pre` signal action throws an exception, the operation is not called and the corresponding `Post` signals are not emitted. Exceptions to this pattern are noted in the table below.



## Signal Types

The naming of `ActivityRegistry` member objects, that are used for emitting the signals, follow a pattern where the first character is in lower case, and `Signal_` is appended to the end of the signal type name.

The order of the signals in this list is the same as in the `ActivityRegistry.h`.

| Signal Type | Are Pre and Post guaranteed to be called in the same thread? | Short explanation |
|-------------|------------------------------------------------------------------|-------------------|
| `PostServicesConstruction` | - | Emitted after all services have been constructed. |
| `{Pre,Post}EventSetupModulesConstruction` | Yes (in the main thread) | Emitted before and after constructing EventSetup modules. |
| `{Pre,Post}ModulesAndSourceConstruction` | Yes (in the main thread) | Emitted before and after concurrent construction of the Source and the EDModules. |
| `{Pre,Post}FinishSchedule` | Yes (in the main thread) | Emitted before and after finalizing the Schedule. |
| `{Pre,Post}PrincipalsCreation` | Yes (in the main thread) | Emitted before and after creating the Principals. |
| `{Pre,Post}ScheduleConsistencyCheck` | Yes (in the main thread) | Emitted before and after checking module dependency correctness and deleting non-consumed unscheduled EDModules. |
| `Preallocate` | - | Emitted with system bounds information for resource preallocation. |
| `{Pre,Post}EventSetupConfigurationFinalized` | Yes (in the main thread) | Emitted before and after finishing the EventSetup configuration. |
| `EventSetupConfiguration` | - | Emitted after the EventSetup Records-to-resolvers indices have been updated. |
| `{Pre,Post}ModulesInitializationFinalized` | Yes (in the main thread) | Emitted before and after finalizing module initialization. |
| `{Pre,Post}BeginJob` | Yes | Emitted before and after running `beginJob()` for all modules. If a `Pre` signal action throws an exception, the modules' `beginJob()` are not called, but the `Post` signal is emitted. |
| `{Pre,Post}EndJob` | Yes | Emitted before and after running `endJob()` for all modules. If a `Pre` signal action throws an exception, the modules' `endJob()` are not called, but the `Post` signal is emitted. |
| `LookupInitializationComplete` | - | Emitted after module paths and consumes relationships have been initialized for EventSetup lookups. The main purpose is to communicate the data dependencies to the Services that are interested of those. |
| `{Pre,Post}BeginStream` | Yes | Emitted before and after running `beginStream()` for all modules. If a `Pre` signal action throws an exception, the modules' `beginStream()` are not called, but the `Post` signal is emitted. |
| `{Pre,Post}EndStream` | Yes | Emitted before and after running `endStream()` for all modules. If a `Pre` signal action throws an exception, the modules' `endStream()` are not called, but the `Post` signal is emitted. |
| `JobFailure` | | _Only called from `FWCore/Services/test/servicesJobReport_t.cpp`, should we remove?_ |
| `{Pre,Post}SourceNextTransition` | Yes | Emitted before and after the input source determines the next transition type (event, run, lumi, etc.). |
| `{Pre,Post}SourceEvent` | Yes | Emitted before and after the `InputSource` makes an `Event`. |
| `{Pre,Post}SourceLumi` | Yes | Emitted before and after the `InputSource` makes a `LuminosityBlock`. |
| `{Pre,Post}SourceRun` | Yes | Emitted before and after the `InputSource` makes a `Run`. |
| `{Pre,Post}SourceProcessBlock` | Yes | Emitted before and after the `InputSource` makes a `ProcessBlock`. |
| `{Pre,Post}OpenFile` | Yes | `PoolSource`, `EmbeddedRootSource`, `RNTupleTempSource`, and `EmbeddedRNTupleTempSource` emit these signals  before and after opening a new input file (either primary or secondary). Note that the first signal may be emitted during the Source construction, and the later signals after the `BeginProcessing`. |
| `{Pre,Post}CloseFile` | Yes | `PoolSource` and `RNTupleTempSource`  emit these signals before and after closing a primary input file file. |
| `{Pre,Post}OpenOutputFiles` | Yes | Emitted before and after opening OutputModules' output files. If a `Pre` signal action throws an exception, the `openOutputFiles()` is not called, but the `Post` signal is emitted. |
| `{Pre,Post}CloseOutputFiles` | Yes | Emitted before and after closing OutputModules' output files. If a `Pre` signal action throws an exception, the `closeOutputFiles()` is not called, but the `Post` signal is emitted. |
| `{Pre,Post}ModuleBeginStream` | Yes | Emitted before and after each module's `beginStream()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module's `beginStream()` is not called, but the `Post` signal is emitted. |
| `{Pre,Post}ModuleEndStream` | Yes | Emitted before and after each module's `endStream()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module's `endStream()` is not called, but the `Post` signal is emitted. |
| `{Pre,Post}BeginProcessBlock` | Yes | Emitted before and after a global `BeginProcessBlock` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}AccessInputProcessBlock` | Yes | Emitted before and after a global `AccessInputProcessBlock` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}EndProcessBlock` | Yes | Emitted before and after a global `EndProcessBlock` transition for all modules. If a `Pre` signal action throws an exception, the  transition is not run, but the `Post` signal is emitted.|
| `BeginProcessing` | - | Emitted right before the data processing phase is started. |
| `EndProcessing` | - | Emitted right after the data processing phase is finished. |
| `{Pre,Post}GlobalBeginRun` | No | Emitted before and after running the global `BeginRun` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}GlobalEndRun` | No | Emitted before and after runninge the global `EndRun` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}WriteProcessBlock` | No | Emitted before and after OutputModules write the data of a ProcessBlock. Exceptions from the `Pre` signal actions are ignored. |
| `{Pre,Post}GlobalWriteRun` | No | Emitted before and after the OutputModules write the data of a Run. Exceptions from the `Pre` signal actions are ignored. |
| `{Pre,Post}StreamBeginRun` | No | Emitted before and after running the stream `BeginRun` transition for all modules on a specific stream. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}StreamEndRun` | No | Emitted before and after running the stream `EndRun`  transition for all modules on a specific stream. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}GlobalBeginLumi` | No | Emitted before and after running the global `BeginLuminosityBlock` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}GlobalEndLumi` | No | Emitted before and after running the global `EndLuminosityBlock` transition for all modules. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}GlobalWriteLumi` | No | Emitted before and after the OutputModules write the data of a LuminosityBock. Exceptions from the `Pre` signal action are ignored. |
| `{Pre,Post}StreamBeginLumi` | No | Emitted before and after running the stream `BeginLuminosityBlock` transition for all modules on a specific stream. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}StreamEndLumi` | No | Emitted before and after running the stream `EndLuminosityBlock` transition for all modules on a specific stream. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}Event` | No | Emitted before and after running the `Event` transition for all modules on a specific stream. If a `Pre` signal action throws an exception, the transition is not run, but the `Post` signal is emitted. |
| `{Pre,Post}ClearEvent` | Yes | Emitted before and after clearing the Event data products after the Event has been processed. |
| `{Pre,Post}PathEvent` | No | Emitted before and after processing a Path for an Event. |
| `PreStreamEarlyTermination` | - | Emitted when an exception is thrown from any stream transition. |
| `PreGlobalEarlyTermination` | - | Emitted when an exception is thrown from any global transition. |
| `PreSourceEarlyTermination` | - | Emitted when an external termination request is received. |
| `{Pre,Post}ESModuleConstruction` | Yes | Emitted before and after constructing an EventSetup module (ESProducer or ESSource). |
| `PostESModuleRegistration` | - | Emitted after an ESModule provider has been registered with the EventSetupProvider. ??? IS THIS CORRECT ??? |
| `ESSyncIOVQueuing` | - | Emitted when an EventSetup IOV synchronization is queued for processing. ??? IS THIS CORRECT ??? |
| `{Pre,Post}ESSyncIOV` | Yes | Emitted before and after synchronizing EventSetup IOVs and setting up EventSetup records. ??? IS THIS CORRECT ???|
| `{Pre,Post}ESModulePrefetching` | No | Emitted before and after prefetching data dependencies for an ESModule. |
| `{Pre,Post}ESModule` | Yes | Emitted before and after running an ESModule's "produce" or `setIntervalFor()` method. |
| `{Pre,Post}ESModuleAcquire` | Yes | Emitted before and after an ESModule's "acquire" method for external work. |
| `{Pre,Post}ModuleConstruction` | Yes | Emitted before and after constructing an EDModule. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleDestruction` | Yes | Emitted before and after an EDModule is destructed because there was no data or control flow dependence on it. |
| `{Pre,Post}ModuleBeginJob` | Yes | Emitted before and after each module's `beginJob()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleEndJob` | Yes | Emitted before and after each module's `endJob()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleEventPrefetching` | No | Emitted before and after each module's prefetching of data products needed by the EDModule for an Event. |
| `{Pre,Post}ModuleEvent` | Yes | Emitted before and after running each EDModule's Event processing method (e.g., `produce()`, `filter()`, `analyze()`). |
| `{Pre,Post}ModuleEventAcquire` | Yes | Emitted before and after each ExternalWork-using EDModule's `acquire()` method. |
| `{Pre,Post}ModuleTransformPrefetching` | No | Emitted before and after each Transformer-using EDModule's prefetching of data products needed by the transformer. |
| `{Pre,Post}ModuleTransform` | Yes | Emitted before and after each Transformer-using EDModules's `transform` function. |
| `{Pre,Post}ModuleTransformAcquiring` | Yes | Emitted before and after each Transformer-using EDModules' `transformAsync`'s "acquire" function. |
| `{Pre,Post}ModuleEventDelayedGet` | Yes | Emitted before and after a module performs a delayed get operation to retrieve a data product. ??? |
| `{Pre,Post}EventReadFromSource` | Yes | `PoolSource`, `EmbeddeRootSource`,  `RNTupleTempSource`, and `EmbeddedRNTupleTempSource` emit these signals before and after reading event provenance and event data products via the delayed reading. In case of prompt reading, the signals are emitted when the data products are retrieved from the cache of the prompt reader. `RepeatingCachedRootSource` emits these signal when retrieving event data products from its cache. When reading the provenance, if the `Pre` signal action throws an exception, the proevenance is not read, but the `Post` signal is called. When reading through delayed reader, if the `Pre` signal action throws an exception, the `Post` signal is not called. |
| `{Pre,Post}ModuleStreamPrefetching` | No | Emitted before and after asynchronously prefetching data products needed by each EDModule for stream transitions (BeginRun, EndRun, BeginLumi, EndLumi). |
| `{Pre,Post}ModuleStreamBeginRun` | Yes | Emitted before and after each EDModule's stream `beginRun()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleStreamEndRun` | Yes | Emitted before and after each EDModule's stream `endRun()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleStreamBeginLumi` | Yes | Emitted before and after each EDModule's stream `beginLuminosityBlock()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleStreamEndLumi` | Yes | Emitted before and after each EDModule's stream `endLuminosityBlock()` method is called on a specific stream. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleBeginProcessBlock` | Yes | Emitted before and after each EDModule's global `beginProcessBlock()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleAccessInputProcessBlock` | Yes | Emitted before and after each EDModule's `accessInputProcessBlock()` method is called to read data from input ProcessBlocks. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleEndProcessBlock` | Yes | Emitted before and after each EDModule's global `endProcessBlock()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleGlobalPrefetching` | No | Emitted before and after asynchronously prefetching data products needed by each EDModule for global transitions (BeginRun, EndRun, BeginLumi, EndLumi). |
| `{Pre,Post}ModuleGlobalBeginRun` | Yes | Emitted before and after each EDModule's global `beginRun()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleGlobalEndRun` | Yes | Emitted before and after each EDModule's global `endRun()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleGlobalBeginLumi` | Yes | Emitted before and after each EDModule's global `beginLuminosityBlock()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleGlobalEndLumi` | Yes | Emitted before and after each EDModule's global `endLuminosityBlock()` method is called. If a `Pre` signal action throws an exception, the module is not constructed, but the `Post` signal is called. |
| `{Pre,Post}ModuleWriteProcessBlock` | Yes | Emitted before and after each OutputModule's `doWriteProcessBlock()` method, that writes ProcessBlock data to output. |
| `{Pre,Post}ModuleWriteRun` | Yes | Emitted before and after each OutputModule's `doWriteRun()` method, that writes Run data to output. |
| `{Pre,Post}ModuleWriteLumi` | Yes | Emitted before and after each OutputModule's `doWriteLuminosityBlock()` method, that writes LuminosityBlock data to output. |
| `{Pre,Post}SourceConstruction` | Yes | Emitted before and after construction of the Source. |
