DQM Services: Core
==================

These packages contain functionality for DQM that is not specific to any data, subsystem or detector. This is the "DQM Framework", provided by the DQM group to allow detector specific code (mostly in `DQM/`, `DQMOffline/` and `Validation/`) to interact with the DQM infrastructure (DQM GUI, Tier0 processing, Online DQM, etc.).

This document contains forward references. Terms may be used before they are defined.


Package Contents
----------------

- `Core/`: Header files used by other DQM modules, implementation of `DQMStore` and `MonitorElement`. No plugins.

The DQM infrastructure
----------------------

_This section explains how different parts of the DQM software work together in general. It is not limited to the `DQMServices/Core` package._

### Which components exist?

#### Plugins

DQM code runs as CMSSW plugins. There are two main types of plugins: *Analyzers* and *Harvesters*. Both exist in a number of technical variations, defined by the base class used.

There are six supported types of DQM modules:
- `DQMEDAnalyzer`, based on `edm::stream::EDProducer`. Used for the majority of histogram filling in RECO jobs.
- `DQMOneEDAnalyzer` based on `edm::one::EDProducer`. Used when begin/end job transitions are required. Can accept more `edm::one` specific options.
- `DQMOneLumiEDAnalyzer` based on `edm::one::EDProducer`. Used when begin/end lumi transitions are needed. Blocks concurrent lumisections.
- `DQMGlobalEDAnalyzer` based on `edm::global::EDProducer`. Used for DQM@HLT and a few random other things. Cannot save per-lumi histograms (this is a conflict with the fact that HLT _typically_ saves _only_ per lumi histograms, see #28341).
- `DQMEDHarvester` based on `edm::one::EDProducer`. Used in harvesting jobs to manipulate histograms in lumi, run, and job transitions. 
- `edm::EDAnalyzer` legacy modules. Can do filling and harvesting. Not safe to use in the presence of concurrent lumisections. Safe for multi-threaded running from the DQM framework side.

The `DQMStore` lives in CMSSW as a `edm::Service` singleton instance.

There are a few special plugins in other `DQMServices/` packages:
- `DQMRootOutputModule`: An output module for the DQMIO format.
- `DQMRootSource`: An input module for the DQMIO format.
- `EDMtoMEConverter`: A module that populates the `DQMStore` from edm products in `MEtoEDM` format, similar to an input module.
- `MEtoEDMConverter` A module that produces `MEtoEDM` products from the `DQMStore` contents, similar to an output module.
- `DQMFileSaver`/`DQMFileSaverOnline`: Modules that save to the legacy `TDirectory` format.
- `DQMFileSaverPB`: A module that saves to the `ProtoBuf` format for `DQM@HLT`.
- `DQMProtoBufReader`: An input module that reads a stream of `ProtoBuf` format files written by `DQMFileSaverPB`.
- `QualityTester`: A special `DQMEDHarvester` that applies the XML-based quality tests.
- `DQMService`: A `edm::Service` singleton that handles the network communication with the DQMGUI in online.


#### File formats

DQM data (mostly histograms, more specifically `MonitorElement`s) can be save in multiple different formats. The formats differ in which features of MEs they can express/persist, and how slow IO is.

- _DQMIO_: this is the "official" DQM data format. It is used for the DQMIO datatier stored after the RECO step in processing jobs. Histograms are stored in ROOT `TTree`s, which makes IO reasonably fast. EDM metadata is preserved, ME metadata is properly encoded and multiple runs/lumisections can be stored in a single file. Some obscure ME options (flags) are not preserved. Reading and writing implemented in EDM input and output modules in `DQMServices/FwkIO`: `DQMRootSource` and `DQMRootOutputModule`.
- _Legacy ROOT_ (`DQM_*.root`): this is the most "popular" DQM data format. It is used for the uploads to DQMGUI and typically saved after the HARVESTING step. Histograms are stored in a `TDirectory`, which makes IO very slow but easy. All ME metadata is preserved (as pseudo-XML `TString`s in the `TDirectory`), but run/lumi metadata is not supported well. Only write support, implemented in `LegacyIOHelper` in `DQMServices/Core`. Only this format supports saving `JOB` histograms.
- _Legacy ProtoBuf (PB)_: this format is similar to legacy ROOT, but instead of the slow `TDirectory`, it uses a Google ProtoBuf based container. ROOT objects are serialized into `TBufferFile` objects, which allow very fast IO but no schema evolution (not suitable for archiving). Used for the `fastHadd` process in DQM@HLT and historically for batch impor/export in DQMGUI. Implemented by `DQMFileSaverPB` and the edm input module `DQMProtobufReader`, in "streamer" format for HLT/DAQ.
- _MEtoEDM/edm Event format_: Using dedicated plugins, DQM histograms can be transformed into EDM products and saved into, or read from, the EDM event files. Causes high memory usage during saving. Full EDM metadata is preserved. This format is used in AlCa workflows between ALCARECO (the data is called ALCAPROMPT) and ALCAHARVEST steps.
- _DQMGUI index format_: Not part of CMSSW. 
- (_DQMNET_): Not a file format. Network protocol used between CMSSW and DQMGUI in online live mode. Implemented by `DQMNet` and `DQMService`. Based on `TBufferFile`. The DQMGUI uses a version of the DQM infrastructure that was frozen at `CMSSW_7_6_0`.

Data is archived for eternity by CMS computing in DQMIO format, in legacy ROOT format by the DQM group (CASTOR tape backups, usually not available for reading, and EOS -- recent data is also available for download in the DQMGUI), and in the DQMGUI index (hosted by CMSWEB, on disk, instant access to any ME).

#### Processing environments

DQM code runs in various places during CMS data processing. How much code runs, and how much processing time is spent doing DQM, varies by orders of magnitude. In rough order of amount of code and time spent, there are

- _RECO_: Processing step in all CMS offline processing (Prompt, Express, ReReco, Monte Carlo, RelVal) where DQM data is extracted from events ("map" in the Map-Reduce pattern). Data is split into an arbitrary number of jobs and processed out of order and massively parallel. Runs `DQM*EDAnalyzer`s, DQMIO output and uses multi-threaded setup of CMSSW (potentially with concurrent lumisections). Configured using the `DQM` or `VALIDATION` steps in `cmsDriver.py`. Typically, only (a fraction of) a single run is processed per job, but jobs processing multiple runs are possible (e.g. in run-dependent Monte Carlo).
- _HARVESTING_: Processing step in all CMS offline processing where data saved by the RECO step in DQMIO format is read and reduced into the final output ("reduce" in the Map-Reduce pattern). Full run data is available. Produces a legacy ROOT output file and uploads it to DQMGUI. Runs `DQMEDHarvester`s and legacy `edm::EDAnalyzer`s, the `DQMRootSource` and `DQMFileSaver`. Harvesting jobs run single-threaded and process all data in order. Typically, only one run is processed per job.
- _DQMIO merge_: Between RECO and HARVESTING, DQMIO files are partially merged into bigger files. No plugins run in this step, only DQMIO input and output. The configuration for these jobs is `Configuration/DataProcessing/python/Merge.py`.
- _ALCARECO_: Like RECO, but on special data and with special code for AlCa. Saves data in `MEtoEDM` format (among others) using the `MEtoEDMConverter`. The DQM-relevant modules are on the `ALCA` step in `cmsDriver.py`.
- _ALCAHARVEST_: Like HARVESTING, but for ALCAPROMPT data in `MEtoEDM` format (produced by ALCARECO jobs), using `EDMtoMEConverter`. No guarantee of in-order processing. Multiple runs may be processed in one job producing one legacy output file (multi-run harvesting).
- _online DQM_: Runs constantly on a small fraction of data directly from HLT while CMS is taking data, on dedicated hardware at Point 5 (4 physical machines). Analyzes a small number of events and live-streams histograms to the online DQMGUI. An identical _playback_ system, also running 24/7 on a small set of stored data, exists as well. Cann run all module types, runs single-threaded and saves legacy `TDirectory` files (using `DQMFileSaverOnline`) in addition to the live output. Uses configuration files from `DQM/Integration/python/clients/` folder. A special case is the online process reading `ProtoBuf` data from `DQM@HLT` using `DQMProtoBufReader` (`hlt_dqm_clientPB`). 
- _DQM@HLT_: A small amount of DQM code runs on all HLT nodes to collect statistics over all events at the input of HLT. These histograms get saved into files in the legacy ProtoBuf format, are merged/reduced using the `fastHadd` utility (part of the CMSSW codebase) on DAQ systems, and streamed into the online DQM system, where they are repacked and continue like normal online DQM. The output is used by a few HLT experts. Only `DQMGlobalEDAnalyzer`s should be used at HLT, and the output is written by `DQMFileSaverPB`. The configuration is manged by HLT (based on confdb).
- _multi-run HARVESTING_: Identical to HARVESTING, except there are multiple runs in a single job. Histograms are accumulated across runs. The only difference to normal HARVESTING is that the output file must not use any run number, but instead use the placeholder `999999`. This is set up in `Configuration/StandardSequences/python/DQMSaverAtJobEnd_cff.py`, which can be activated using the `cmsDriver.py` option `--harvesting AtJobEnd`.
- _commissioning tools_ and other private workflows: These are manually built and run configurations. They can use any type of DQM modules (commonly including legacy modules) and typically write `TDirectory` output using a `DQMFileSaver` at the end of the job. It is also possible that another module calls `DQMStore::save` to write a `TDirectory` based output file. 

#### Library classes

The DQM framework is implemented in a number of classes, some of which can be used directly by subsystem code (in special cases).

There are five header files available for subsystem code:
- `DQMEDAnalyzer.h, DQMOneEDAnalyzer.h, DQMGlobalEDAnalyzer.h, DQMEDHarvester.h` for modules of the respective type.
    - These must be included _before_ any other files for technical reasons: `DQMEDAnalyzer.h` contains some template logic that must be declared before the EDM headers are loaded.
- `DQMStore.h` for any other usages.
    - It includes all other DQM header files, primarily `MonitorElement.h`.
    - Other header files might be removed in the future.

##### The DQM namespaces.

The DQM classes come in three variations, which are exposed in three different namespaces: `dqm::reco`, `dqm::harvesting`, and `dqm::legacy`. Each contains the same classes, but the exposed interface vary slightly. Currently, `dqm::harvesting` and `dqm::legacy` are identical, but `dqm::reco` provides less operations on the `MonitorElement`. The common implementation is in the internal namespace `dqm::implementation`. There are inheritance relations between the different variations to allow implicit conversions in some common cases.

Usually, the `dqm::` namespaces should not be visible in DQM module code: The module base classes each import the correct classes into their local namespace, so they can be used directly as `DQMStore` and `MonitorElement`. However, for legacy modules, definitions outside the class scope (like needed with `DQMGlobalEDAnalyzer`), and return values of methods (which are declared outside class scope), it may be required to specify the correct namespace.


##### The `DQMStore`.

All operations on the `DQMStore` should take a single lock (when required), to make all interactions with the internal data structures thread safe. This is a recursive mutex for simplicity: The same global lock is held while subsystem booking code is running, to prevent issues when non-thread-safe operations (e.g. on `TH1` objects) are done in booking. This "outer" layer of locking can be removed if all booking code is thread-safe. To achieve this, all operations on bare ROOT objects outside the `MonitorElement` need to be protected, e.g. using the callback function that is available in booking calls.

Actions on the `DQMStore` are usually indirect and done via the `IBooker` and `IGetter` interfaces. The `DQMStore` implements these interfaces itself, but it is possible to instantiate stand-alone `IBooker` and `IGetter` instances that keep their own "session" state but forward all operations to a different `DQMStore`. The filesystem-like parts of the stateful interface (`cd`, `pwd`, etc.) are provided by a common base class `NavigatorBase`. Note that this interface can be quite surprising in it's behaviour, since it is kept bug-for-bug compatible with previous versions. New code should only use `setCurrentFolder`.

The behavior of the `DQMStore` is not affected by any configuration options (this differs from earlier implementations, where multiple _modes_ existed). Yet, there are a few options that can be set:
- There are two options to control per-lumi saving (`saveByLumi` in the `DQMStore`) and harvesting (`reScope` in `DQMRootSource`). Both can be expressed in terms of _Scope_, see later. These options do not affect what the `DQMStore` itself does: `saveByLumi` affects the default scope and is passed down to the booking code in the DQM modules, while `reScope` is handled in the input sources.
- Another option, `assertLegacySafe`, only adds assertions to make sure no operations that would be unsafe in the presence of legacy modules sneak in. It does not affect the behaviour. It is disabled by default and indeed it should be kept disabled in production RECO jobs where concurrent lumisections are expected.
- The `verbose` option should not affect the behavior. Setting it to a value of at least 4 will enable printing a backtrace for every booking call.
- The `trackME` option is a tool for debugging. When it is set, the `DQMStore` will log all life cycle events affecting MEs matching this name. This does not include things done to the ME (like filling -- the `DQMStore` is not involved there), but it does include creation, reset, recycling, and saving of MEs.
    
##### The `MonitorElement`.

There is a related class, `MonitorElementData`, which is located in `DataFormats/Histograms`. This class holds all the state of the `MonitorElement`. This split is motivated by the idea of actually storing the MEs in EDM products; however this is not done in the current implementation, since EDM requires products to be immutable while the current harvesting API assumes mutable objects everywhere. This could be worked around by using a copy-on-write approach, but this turned out rather complicated for very small benefits and was abandoned.

- There is (effectively) only one type of ME, and it uses locking on a local lock to allow safe access from multiple threads.
- There are three different namespaces (`reco`, `harvesting`, `legacy`) providing this ME type, however the `reco` version does not expose some non-thread-safe APIs.
    - However, it also _does_ still expose some of the APIs which allow direct access to ROOT objects, which is _not_ thread safe! These should be removed as soon as possible, but there is still code that relies on these APIs.
- The `MonitorElement` can _own_ the `MonitorElementData` that holds the actual histogram (that is the case for _global_ MEs), or _share_ it with one ME that owns the data and others that don't (this is the case for _local_ MEs).
    - Even though we talk about _local_ and _global_ MEs (see later), they are represented by the same class and only the ownership of the `MonitorElementData` differs.
- The `MonitorElementData` does not provide an API. It is always wrapped in a `MonitorElement`.
- The `MonitorElement` has _no_ state, apart from the pointer to the data and the `is_owned` flag.
    - Actually, it does have some state in the `DQMNet::CoreObject`, which is present in the `MonitorElement` to remain compatible with `DQMNet` for online DQM. The values there (dir, name, flags, qtests) are to be considered cached copies of the "real" values in `MonitorElementData`. The dir/name copy in the `CoreObject` is also used as a set index for the `std::set`s in the `DQMStore`, since it remains available even when the `MonitorElementData` is detached from local MEs (e.g. between lumisections).
- The `MonitorElement` still allows access to the underlying ROOT object (`getTH1()`), but this is unsafe and should be avoided whenever possible. However, there is no universal replacement yet, and DQM framework code uses direct access to the ROOT object in many places.

##### The Quality Tests.

Various types related to quality tests exist in various namespaces (`dqm::qstatus`, `me_util`, `DQMChannel`, `QReport` etc.). These definitions are kept to provide compatibility to some existing code.

The main declarations are part of the `MonitorElementData` (`QReport` which contains `QValue` and `DQMChannel`). The quality tests themselves are subclasses of `QCriterion`, defined in `QTest.h`. All of the quality test handling is centralized in the `QualityTester` module, with only the code required to provide all the APIs in the `MonitorElement`. There is some duplication the quality test related members of the `MonitorElement` to provide the interface required for the `DQMNet` code, that was kept from an older implementation.

##### The `DQMService` and `DQMNet`.

These classes are only used in online DQM, to provide live streaming of the DQM histograms to the online DQM GUI. The `DQMService` is an `edm::Service` (not to be confused with the `DQMStore` `edm::Service`!), that runs the `DQMNet` network client in a dedicated thread. This code was kept unchanged since a long time, as it works reliably. Consequently, it does use some unusual APIs which are only kept to support this code.

### How do the components interact?

Each plugin has a pointer to the single, global `DQMStore` instance. This pointer is either obtained via the `edm::Service` mechanism (legacy) or passed in from the base class in the form of `IBooker`/`IGetter` objects. (The base classes still obtain them via `edm::Service`). Since the `DQMStore` contains global, mutable state (not only the histogram storage itself, but also e.g. the stateful booking interface with `cd`), interaction with the `DQMStore` is not fully thread safe: while concurrent accesses are _safe_, the results might not make sense. 

To monitor a quantity, the plugin _books_ a `MonitorElement` with the `DQMStore`. This can happen at any time (legacy code) or only in a specific callback from the base class (`bookHistograms` in `DQM*EDAnalyzer`), which is called once per run. The `MonitorElement` is identified by a path that should be unique. The path can be used by other plugins to get access to the same `MonitorElement` (though in most cases, this is not intended!). The result of the booking call is a bare pointer to the `MonitorElement`.

The `MonitorElement` can now be filled with data using the `Fill` call, similar to a ROOT histogram. This can happen at any time. the `MonitorElement` is implemented as a thin wrapper with a lock around a ROOT object. During booking, the ROOT object may be accessed directly to configure the histogram (axis, labels, fill modes), however such things should happen in a callback function passed into the booking call to guarantee that the customization code is run only once and that the operation is atomic with respect to other modules accessing the same ME. 

When a ME is booked, internally  _global_ and _local_ MEs are created. This should be invisible to the user; the technical details are as follows:
- In the DQM API, we face the conflict that `MonitorElement` objects are held in the modules (so their life cycle has to match that of the module) but also represent histograms whose life cycle depends the data processed (run and lumi transitions). This caused conflicts since the introduction of multi-threading.
- The `DQMStore` resolves this conflict by representing each monitor element using (at least) two objects: A _local_ `MonitorElement`, that follows the module life cycle but does not own data, and a _global_ `MonitorElement` that owns histogram data but does not belong to any module. There may be multiple _local_ MEs for one _global_ ME if multiple modules fill the same histogram (`edm::stream` or even independent modules). There may be multiple _global_ MEs for the same histogram if there are concurrent lumisections.
- The live cycle of _local_ MEs is driven by callbacks from each of the module base classes (`enterLumi`, `leaveLumi`). For legacy `edm::EDAnalyzer`s, global begin/end run/lumi hooks are used, which only work as long as there are no concurrent lumisections. The _local_ MEs are kept in a set of containers indexed by the `moduleID`, with special value `0` for _all_ legacy modules and special values for `DQMGlobalEDAnalyzer`s, where the local MEs need to match the life cycle of the `runCache` (module id + run number), and `DQMEDAnalyzer`s, where the `streamID` is combined with the `moduleID` to get a unique identifier for each stream instance.
- The live cycle of _global_ MEs  is driven by the `initLumi/cleanupLumi` hooks called via the edm service interface. They are kept in a set of containers indexed by run and lumi number. For `RUN` MEs, the lumi number is 0; for `JOB` MEs, run and lumi are zero. The special pair `(0,0)` is also used for _prototypes_: Global MEs that are not currently associated to any run or lumi, but can (and _have to_, for the legacy guarantees) be recycled once a run or lumi starts. 
- If there are no concurrent lumisections, both _local_ and _global_ MEs live for the entire job and are always connected in the same way, which means all legacy interactions continue to work. `assertLegacySafe` (enabled by default) checks for this condition and crashes the job if it is violated.


Whenever the DQM data should be copied somewhere else (output file or live monitoring), a plugin queries the `DQMStore` for `MonitorElement`s. These calls will return _global_ MEs, and since there can be multiple _global_ MEs for different lumisections, such modules need to be aware of concurrent lumisections. The majority of the interfaces to read MEs from the `DQMStore`, which are provided in the `IGetter` interface, will return _any_ global ME, and can only work correctly in jobs that run sequentially (this is the case for harvesting jobs, where these interfaces are primarily used).

When reading histograms from DQMIO  or MEtoEDM data for merging or harvesting, matching histograms from different files need to be merged. As long as no more than a single run is covered and the data was produced using a sane configuration (same software version for all files), this should always be possible, since the set of histograms booked cannot change within a run. However, in multi-run harvesting, it is possible that histograms of the same path are not booked with the same parameters and cannot be merged. The merging code tries to catch and ignore these cases, but it can still fail and crash in certain scenarios (e.g. sometimes ROOT fails merging even on identical histograms). The merging is all handled by the input modules (`DQMRootSource` and `EDMtoMEConverter`) and can be controlled by the `reScope` option.

To clarify how (per-lumi) saving and merging of histograms happens, we introduce the concept of _Scope_. For most DQM code, the default Scope setting will be sufficient, but setting Scope explicitly can be useful in some cases.
- The _scope_ of a ME can be one of `JOB`, `RUN`, or `LUMI`.
    - There is space for a scope between `RUN` and `LUMI`, e.g. for blocks of ten lumisections. Such a scope could be implemented, but no such code exists so far.
- The _scope_ defines how long a _global_ ME should be used before it is saved and replaced with a new histogram.
- The _scope_ must be set during booking.
- By default, the _scope_ for MEs is `RUN`.
    - Code can explicitly use `IBooker::setScope()` to change the scope to e.g. `LUMI`. This replaces the old `setLumiFlag`.
    - To set the scope for a few MEs, there is a RAII scope guard (unrelated concept) to temporarily change the Scope: `IBooker::UseLumiScope`, `IBooker::UseRunScope`, `IBooker::UseJobScope`.
    - When the `saveByLumi` option in the `DQMStore` is set, the default scope changes to `LUMI` for all modules that can support per-lumi saving (`DQMEDAnalyzer` and `DQMOneLumiEDAnalyzer`). It could still be manually overridden in code.
    - In harvesting, the default scope is `JOB`. This works for single-run as well as multi-run harvesting. Moving to scope `RUN` for non-multi-run harvesting would be cleaner, but requires bigger changes to existing code.
    - For legacy modules, the default scope is always `JOB`. This is similar to the old behaviour and ensures that there is only a single, _global_ ME that will be handed out to the legacy module directly.
- When harvesting, we expect histograms to get merged. This merging can be controlled using a single option in `DQMRootSource`: `reScope`. This option sets the _finest allowed scope_ when reading histograms from the input files.
    - When just merging files (e.g. in the DQMIO merge jobs), `reScope` is set to `LUMI`. The scope of MEs is not changed, histograms are only merged if a run is split over multiple files.
    - When harvesting, `reScope` is set to `RUN` (or `JOB`). Now, MEs saved with scope `LUMI` will be switched to scope `RUN` (or `JOB`) and merged. The harvesting modules can observe increasing statistics in the histogram as a run is processed (like in online DQM).
    - For multi-run harvesting, `reScope` is set to `JOB`. Now, even `RUN` histograms are merged. This is the default, since it also works for today's single-run harvesting jobs.
- Currently, EDM does not allow `JOB` products, and therefore output modules cannot save at the end of the `JOB` scope. The only file format where `JOB` scope is supported is the legacy `TDirectory` `DQMFileSaver`.
    - We use `ProcessBlock` products to model the dataflow that traditionally happened at `endJob`.
    - Legacy modules cannot do `JOB` level harvesting: their code in `endJob` only runs after the output file is saved. Some code remains there, and it can still work if output is saved by a different means than `DQMFileSaver`.
    - This means that harvesting jobs can _only_ save in legacy format, since most harvesting logic runs at `dqmEndJob` (a.k.a. `endProcessBlock`. Output modules don't support `ProcessBlock`s yet.
    - For the same reason, _all_ harvesting jobs use `reScope = JOB` today. However, for single-run harvesting jobs, some logic in the `DQMFileSaver` still attaches the run number of the (only) run processed to the output file (this can and will go wrong if there is more than one run in a "normal" harvesting job).
    - This also means that apart from the run number on the output file, "normal" and multi-run harvesting jobs are identical.

Harvesting jobs are always processed sequentially, like the data was taken: runs and lumisections are processed in increasing order. This is implemented in `DQMRootSource`.

DQM promises that all data dependencies across the `DQMStore` are visible to EDM. To achieve this in the major job setups (RECO and HARVESTING), there are multiple generations of _token_ products (`DQMToken`). These products do not hold any data; they only model dependency. The data is kept in the `DQMStore`.

- The tokens are divided into three _generations_: `DQMGenerationReco`, `DQMGenerationHarvesting`, `DQMGenerationQTest`, denoted in the instance label.
- `DQMGenerationReco` is produced by `DQMEDAnalyzer`s, which consume only "normal" (non-DQM) products.
- `DQMGenerationHarvesting` is produced by `DQMEDHarvester`s, which consume (by default) all `DQMGenerationReco` tokens. This allows all old code to work without explicitly declaring dependencies. `DQMEDHarvester`s provide a mechanism to consume more specific tokens, including `DQMGenerationHarvesting` tokens from other harvesters.
- `DQMGenerationQTest` is produced only by the `QualityTester` module (of which we typically have many instances). The `QualityTester` has fully configurable dependencies to allow the more complicated setups sometimes required in harvesting, but typically consumes `DQMGenerationHarvesting`.
- There is a hidden dependency between end run and end process block in harvesting: Many harvesters effectively depend on `DQMGenerationQTest` products (the QTest results) and do not specify that, but things still work because they do their work in `dqmEndJob` (a.k.a. `endProcessBlock`).

#### Local `MonitorElement`s vs. global `MonitorElement`s

The distiction between _global_ and _local_ MEs is the key part to having `edm::stream` based processing in DQM. This distinction bridges the gap between the module's view (`MonitorElement`s are objects that can be used at any time) and the frameworks view (`MonitorElement`s are datastructures that hold data for one run/lumisection/job).

To do this, we have _local_ MEs, which are owned by the modules, and remain valid over the full lifetime of a module (that will be a full job). These are the objects that normal `book*` calls return. Internally, we handle _global_ MEs, which are only valid over a single run or lumisection or job: They are created at the beginning, then filled, then saved to an output file, then destroyed. Modules *can* handle global MEs as well, they are returned by the `get*` calls in the `IGetter` interface of the `DQMStore`. (`book*` calls in _legacy_  and harvesting modules, that is those booking calls that do not happen inside a _booking transaction_ and therefore are booked with module id 0, also return global MEs. Local MEs are still internally created and could be returned as well, but by returning the global objects we get the useful property that `book*` and `get*` for the same name return _the same_ object.)

Global and local MEs use the same type (or rather types), `MonitorElement`. This is questionable and may be slightly confusing, but it keeps things simpler in harvesting code. The difference between global and local MEs is that the global MEs *own* a histogram (in the form of a `MonitorElementData` instance), while the local MEs only *have access* to the histogram, which is owned by the global ME.

Effectively, each local ME is connected to a global ME, by sharing its `MonitorElementData`. There can be many local MEs sharing data of the same global ME (this happens e.g. with `edm::stream` modules, and this is why we need locking in the `MonitorElement` methods). There can also be multiple global MEs with the same name, e.g. if there are concurrent lumisections. In this case, the local MEs must be connected to the *correct* global ME. This is handled by the `enterLumi` method. It can happen that a local ME is not connected to *any* global ME, accessing it in this situation will lead to a crash. This should only happen during transitions.

Related to multi-threading and legacy modules, we guarantee two things: First, it is safe to access _local_ MEs during the callbacks of `DQM*EDAnalyzer`. This is true no matter how many threads and concurrent accesses there are. Second, if a job runs single-threaded and without concurrent lumisections, then local and global MEs are interchangeable (though not identical) and valid for the entire job. There is exactly one global ME for each name, over the entire job, and all local MEs for the same name are always attached to this global ME. This is achieved by _recycling_ run and lumi MEs (that is, the global MEs for a new run/lumi will re-use the already existing objects, so that existing pointers remain valid) and carefully placed `Reset` calls, so that even `fill` calls that reach the MEs before or between runs/lumisections will be saved in the next run/lumi. Those guarantees cannot be held when there are multiple threads (modules on different threads might see invalid states during transitons) or concurrent lumisections (per-lumi global MEs may be created and deleted during the job).

### Digging Deeper

For more details about how things are exactly implemented, refer to the comments in the source code; primarily `DQMStore.h`, `DQMStore.cc`, `MonitorElement.h` and `MonitorElement.cc`. For hints on how to use the APIs, refer to the sample and test code in `DQMServices/Demo`. The module header files `DQM*EDAnalyzer.h`, `DQMEDHarvester.h` can also be worth a look, especially to understand the interaction with EDM.

To actually unterstand how MEs are handled by the `DQMStore`, the `trackME` feature can be very useful:
To understand how a ME is handled, set the `process.DQMStore.trackME = cms.untracked.string("<ME Name>")` option in the config file.
- The `DQMStore` will then log all life cycle events affecting MEs matching this name. This does not include things done to the ME (like filling -- the `DQMStore` is not involved there), but it does include creation, reset, recycling, and saving of MEs.
- The matching is a sub-string match on the full path, so it is also possible to watch folders or groups of MEs.
- For the more difficult cases, it can make sense to put a debug breakpoint (`std::raise(SIGNINT)`) inside `DQMStore::debugTrackME` to inspect the stack when a certain ME is created/modified.
- The old functionality of logging the caller for all booking calls also still exists and can be enabled by setting `process.DQMStore.verbose = 4`.
