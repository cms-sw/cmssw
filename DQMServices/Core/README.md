DQM Services
============

These packages contain functionality for DQM that is not specific to any data, subsystem or detector. This is the "DQM Framework", provided by the DQM group to allow detector specific code (mostly in `DQM`, `DQMOffline/` and `Validation/` to interact with the DQM infrastructure (DQM GUI, Tier0 processing, Online DQM, etc.).

This document contains forward references. Terms may be used before they are defined.


Package Contents
----------------

- `Components/`: Collection of (independent) DQM plugins that handle core functionality. This includes
    - `QualityTester`: Applies Quality Tests defined in XML configuration to MEs
    - `DQMDaqInfo`: TODO: ?
    - `DQMEventInfo`: TODO: ?
    - `DQMFEDIntegrityClient`: TODO: ?
    - `DQMMessageLogger`: Creates histograms of number of EDM error messages.
    - `DQMMessageLoggerClient`: Some sort of post-processing on these histograms.
    - `DQMFileSaver`: Triggers legacy format saving of DQM output in harvesting jobs.
    - `DQMLumiMonitor`: Some sort of SiPixel based luminosity measurement. Does not belong into this package. Used online only.
    - `DQMScalInfo`: Reports some data from Lumi/L1T Scalers. Used online.
    - `DQMProvInfo`: Populates the `Info/*Info` histograms.
    - `DQMStoreStats`: Provides some DQM self-monitoring (memory etc.) Not used in production.
    - `EDMtoMEConverter`: Reads histograms from EDM event files and puts them into DQM. Used for AlCa.
    - `MEtoEDMConverter`: Reads histograms from the DQMStore and saves them into EDM products. Used for AlCa.
    - `python/`: TODO: some sort of web interface for something?
    - `scripts/`: Tools to acces DQMGUI and to compare/inspect DQM data files.
    - `test/`: DQM unit tests.
- `Core/`: Header files used by other DQM modules, implementation of `DQMStore` and `MonitorElement`. No plugins.
- `FwkIO/`: The DQMIO input and output modules.
- `FileIO/`: More modules triggering saving to legacy files, for online and HLT. See also: `Components/DQMFileSaver`.
- `StreamerIO/`: Raw data input for online, plus general online DQM tools. Includes the bridge to `DQM^2` (`esMonitoring.py`)



Historical: DQM before the 2019 migration
-----------------------------------------

_a.k.a. 2014/2015 threaded DQM_

_This is information about how DQM used to work. Even though it no longer works that way, and some of the things mentioned don't exist any more, lots of terminology and comments refer to this old design._

### Which components exist?

#### Plugins

DQM code runs as CMSSW plugins there are two main types of plugins: *Analyzers* and *Harvesters*. Both exist in a number of technical variations, defined by the base class used.

- Analyzers:
    - `DQMEDAnalyzer`: Based on `edm::stream::EDAnalyzer` (2015) or `edm::one::EDProducer` (2018), the recommended default base class.
    - `one::DQMEDAnalyzer`: Based on `edm::one`, to be used if `edm::one` behaviour is specifically required.
        - `DQMRunEDProducer` is used if the template parameters indicate that *no* lumi-based histograms are needed.
        - `DQMLumisEDProducer` is used if the template parameters indicate that lumi-based histograms are needed.
    - `DQMGlobalEDAnalyzer`: largely independent `edm::global::EDAnalyzer`. Only recommended for specific applications (DQM@HLT).
    - `edm::EDAnalyzer`: EDM legacy base class, still widely used.
- Harvesters:
    - `DQMEDHarvester`: Recommended base class for harvesters, `edm::one::EDProducer` based.
    - `edm::EDAnalyzer`: EDM legacy base class, still widely used.

The `DQMStore` lives in CMSSW as a `edm::Service` singleton instance. For DQMIO, there is an `OutputModule` (`DQMRootOutputModule`), and an `InputSource` (`DQMRootSource`). 

#### Library classes

Various functionalities are provided in classes which can be used by the CMSSW plugins. The main ones are

- `DQMStore`, to interface with the `DQMStore` service and register (book) `MonitorElement`s with the framwork.
- `MonitorElement`, an object representing a single histogram (or scalar value) of a monitored quantity. In practise, this is a thin wrapper around a ROOT `TH1` instance.
- `ConcurrentMonitorElement` is a wrapper around `MonitorElement` used by `DQMGlobalEDAnalyzer`.
- `QCriterion`, an automated check/comparison on a histogram. Many subclasses for specific checks exist.

#### File formats

DQM data (mostly histograms, more specifically `MonitorElement`s) can be save in multiple different formats. The formats differ in which features of MEs they can express/persist, and how slow IO is.

- _DQMIO_: this is the "official" DQM data format. It is used for the DQMIO datatier stored after the RECO step in processing jobs. Histograms are stored in ROOT `TTree`s, which makes IO reasonably fast. EDM metadata is preserved, ME metadata is properly encoded and multiple runs/lumisections can be store in a single file. Obscure ME options (reference histograms, flags, etc.) are not preserved.
- _Legacy ROOT_ (`DQM_*.root`): this is the most "popular" DQM data format. It is used for the uploads to DQMGUI and typically saved after the HARVESTING step. Histograms are stored in a `TDirectory`, which makes IO very slow but easy. All ME metadata is preserved (as pseudo-XML `TString`s in the `TDirectory`), but run/lumi metadata is not supported everywhere.
- _Legacy ProtoBuf (PB)_: this format is similar to legacy ROOT, but instead of the slow `TDirectory`, it uses a Google ProtoBuf based container. ROOT objects are serialized into `TBufferFile` objects, which allow very fast IO but no schema evolution (not suitable for archiving). Used for the `fastHadd` process in DQM@HLT and historically for batch impor/export in DQMGUI.
- _MEtoEDM/edm Event format_: Using dedicated plugins, DQM histograms can be transformed into EDM products and saved into, or read from, the EDM event files. Support for obscure ME features is unknown, IO performance is also unknown, full EDM metadata is preserved. This format is used in AlCa workflows between ALCARECO and ALCAHARVEST steps (TODO: confirm, very little is know about this usage).
- _DQMGUI index format_: Not part of CMSSW. 
- (_DQMNET_): Not a file format. Network protocol used between CMSSW and DQMGUI in online live mode. Based on `TBufferFile`.

Data is archived for eternity by CMS computing in DQMIO format (TODO: confirm the precise rules), in legacy ROOT format by the DQM group (CASTOR tape backups, usually not available for reading), and in the DQMGUI index (hosted by CMSWEB, on disk, instant access to any ME).

#### Processing environments

DQM code runs in various places during CMS data processing. How much code runs, and how much processing time is spent doing DQM, varies by orders of magnitude. In rough order of amount of code and time spent, there are

- _RECO_: Processing step in all CMS offline processing (Prompt, Express, ReReco, Monte Carlo, RelVal) where DQM data is extracted from events ("map" in the Map-Reduce pattern). Data is split into an arbitrary number of jobs and processed massively parallel.
- _HARVESTING_: Processing step in all CMS offline processing where data saved by the RECO step in DQMIO format is read and reduced into the final output ("reduce" in the Map-Reduce pattern). Full run data is available. Produces a legacy ROOT output file and uploads it to DQMGUI.
- _ALCARECO_: Like RECO, but on specical data and with special code for AlCa. Saves data in MEtoEDM format (among others?).
- _ALCAHARVEST_: Like HARVESTING, but for ALCARECO data.
- _online DQM_: Runs constantly on a small fraction of data directly from HLT while CMS is taking data, on dedicated hardware at Point 5 (4 physical machines). Analyzes a small number of events and live-streams histograms to the online DQMGUI. An identical _playback_ system, also running 24/7 on a small set of stored data, exists as well. 
- _DQM@HLT_: A small amount of DQM code runs on all HLT nodes to collect statistics over all events at the input of HLT. These histograms get saved into files in the legacy ProtoBuf format, are merged/reduced using the `fastHadd` utility on DAQ systems, and streamed into the online DQM system, where they are repacked and continue like normal online DQM. The output is used by a few HLT experts.
- _DQMIO merge_: Between RECO and HARVESTING, DQMIO files are partially merged into bigger files. No plugins run in this step, only DQMIO input and output.

Other use cases of the DQM infrastructure exist (_private workflows_), and include for example _multi-run harvesting_, _commissioning tools_ and _validation tools_. For some of them, configuration files are present in CMSSW, but they are not run automatically on production infrastructure.

#### DQMStore modes

The behaviour of the DQMStore serivce can be affected by a number of _modes_ that can be set in the configuration. Which combination of flags exactly enables which mode is not entirely clear.

- _legacy mode_: When `enableMultiThread` is `false`, the `DQMStore` operates in legacy mode, which means it behaves as a single collection of histograms with names. When `enableMultiThread` is `true`, each module and stream get their own, independent view on the histograms. 

- _collate histograms_: When `collateHistograms_` is set to `true`, histograms are not reset at the end of the run/when being booked again. Else, they are.

- _ls based mode_: When `LSbasedMode` is set to `true`, all histograms are saved every lumisection, else only those with the `lumiFlag` set.

- _force reset on begin lumi_: When `forceResetOnBeginLumi` is set, all histograms with the `lumiFlag` set are automatically reset at the beginning of the lumisection.

### Modes of DQM operation

The components and modes mentioned in the previous section can technically be combined in arbitrary ways. However, for most combinations, the behaviour is unknown and probably not useful. Some known combinations are:

- Running in the _RECO_ step (or similar), with all modes off, and only `DQMEDAnalyzer` plugins (default, one, or global). 
    - Legacy `edm::EDAnalyzer` modules und harvesters are not allowed/supported, though some run and work fine.
- Running in _HARVESTING_ (or similar), with _legacy mode_, and `DQMEDHarvester` modules. Legacy `edm::EDAnalyzer` modules are allowed.
    - `DQMEDAnalyzer`s are technically not supported, but may work fine.
- Running in _HARVESTING_, with _legacy mode_ and _collate histograms_ set. This is used for _multi-run harvesting_.
- Running in _online DQM_, with _legacy mode_ and _ls based mode_ set. This is used for most of online DQM.
- Running in the _DQMIO merge step_, with _force reset on begin lumi_ set.
    - No plugins run in this configuration.
- Running with _legacy mode_ set. This is used for most other configurations.

Notice that there is no way to run harvesters without setting _legacy mode_: The point of multi-threaded mode is to prevent communication between plugins, and harvesters typically need to read MEs produced by other plugins. A thread-safe mechanism to exchange data was never introduced.

### How do the components interact?

Each plugin has a pointer to the single, global `DQMStore` instance. This pointer is either obtained via the `edm::Service` mechanism (legacy) or passed in from the base class in the form of `IBooker`/`IGetter` objects. (The base classes still obtain them via `edm::Service`). Since the `DQMStore` contains global, mutable state (not only the histogram storage itself, but also e.g. the stateful booking interface with `cd`), interaction with the `DQMStore` is never thread safe. Unsafe, multi-threaded access is possible when the `edm::Service` is used to access the instance, but mostly prevented by convervative locking around the booking callbacks and legacy modules. 

To monitor a quantity, the plugin _books_ a `MonitorElement` with the `DQMStore`. This can happen at any time (legacy code) or only in a specific callback from the base class (`bookHistograms` in `DQMEDAnalyzer`), which is called once per run. The `MonitorElement` is identified by a path that should be unique, together with run number, lumi number (if by lumi), plugin (module) id and stream id (before 2018). The path can be used by other plugins to get access to the same `MonitorElement` (only in legacy mode, or when _global_ `MonitorElement`s are explicitly requested). The result of the booking call is a bare pointer to the `MonitorElement`.

The `MonitorElement` can now be filled with data using the `Fill` call, similar to a ROOT histogram. This can happen at any time. the `MonitorElement` is implemented as a thin wrapper around a ROOT object and all ROOT APIs can be accessed as well. This is usually used in booking to configure the histogram (axis, labels, fill modes). Interactions with the `MonitorElement` are also not thread safe, however `MonitorElement`s are typically local to the plugin instance, and can be safely used in all but `edm::global` modules. The `DQMStore` in threaded mode enforces that pointers to a `MonitorElement` instance are only handed out to a single plugin instance, in case of `edm::stream` modules, multiple `MonitorElement` instances are booked.

For use in `edm::global` modules, `ConcurrentMonitorElement` provides a (partially) thread-safe, but incompatible, API. Otherwise, `ConcurrentMonitorElement`s are backed by normal `MonitorElement`s and handled the same.

Whenever the DQM data should be copied somewhere else (output file or live monitoring), a plugin queries the `DQMStore` for `MonitorElement`s. In threaded mode, the `MonitorElement`s need to be cloned under certain circumstances, because the plugin "owning" a `MonitorElement` may continue filling it from another thread while it is used. 

`MonitorElement`s typically collect statistics over a run or job (depending on _collate mode_, though typically irrelevant because there is only one run per job -- many modules make assumptions that are incorrect with more than one run per job). In the _RECO_ case, jobs are typically a small fraction of a run, and full run statistics are only available in _HARVESTING_. To get histograms saved on a finer granularity, _ls based mode_ (global) or the _lumi flag_ (per ME) can be used. This will make sure that the ME is saved every lumisection, and in HARVESTING statistics _accumulate_ over the run, unless the histogram is explicitly reset. Bugs and incorrect assumptions are common regarding this behaviour.

When reading histograms from DQMIO data for merging or harvesting, matching histograms from different files need to be merged. As long as no more than a single run is covered and the data was produced using a sane configuration (same software version for all files), this should always be possible. However, in multi-run harvesting, it is possible that histograms of the same path are not booked with the same parameters and cannot be merged. The merging code tries to catch and ignore these cases, but it can still fail and crash in certain scenarios (e.g. sometimes ROOT fails merging even on identical histograms).

Planned Future: DQM after the 2019 ("product") migration
------------------------------------------------

_a.k.a. The new DQMStore project._

### Which components exist?

#### Plugins

DQM code runs as CMSSW plugins there are two main types of plugins: *Analyzers* and *Harvesters*. Both exist in a number of technical variations, defined by the base class used.

- Analyzers:
    - `DQMEDAnalyzer`: Based on `edm::stream::EDProducer`, the recommended default base class.
    - `DQMOneEDAnalyzer`: Based on `edm::one::EDProducer`, to be used if `edm::one` behaviour is specifically required. Limited functionality, only run-based histograms are possible.
    - `DQMOneLumiEDAnalyzer`: Based on `edm::one::EDProducer` with lumi transitions. Not recommended for future use, only to migrate existing legacy code.
    - `DQMGlobalEDAnalyzer`: Based on `edm::global::EDProducer`, only recommended if `edm::global` behaviour is really required. Limited functionality, only run-based histograms are possible.
    - `edm::EDAnalyzer`: EDM legacy base class. Can be safely used but will by default not interact with the rest of DQM.
- Harvesters:
    - `DQMEDHarvester`: Recommended base class for harvesters, `edm::one::EDProducer` based.

For DQMIO, there is an `OutputModule` (`DQMRootOutputModule`), and an `InputSource` (`DQMRootSource`). 

#### Library Classes

- `DQMStore` is the main container that manages `MonitorElement`s. It does not interact with the outside world and can be instantiated as needed. The plugin base classes all provide a `dqmstore_` member, and the base classes make sure that all histograms in this `DQMStore` are visible to the rest of DQM.
- `MonitorElement` is a thread-safe proxy for a histogram. `MonitorElement` provides all the histogram operations that are commonly used and forwards them to its backing object, `MonitorElementData`. Multiple `MonitorElement`s can share the same `MonitorElementData`. Multi-threaded access is memory-safe, but race conditions and order dependence can appear when non-commutative operations are used (that means any operation except `Fill`).

These two classes (plus the `IBooker` and `IGetter` interfaces referring to/used by them) exist in three different namespaces:
- `dqm::legacy::` specifies the full interface, mostly as provided by the old `MonitorElement`. Any `MonitorElement` can be (implicitly) casted to `dqm::legacy::MonitorElement`, how ever operations on legacy `MonitorElement`s can fail at runtime.
- `dqm::reco::` restricts the interface to operations that are safe in a multi-threaded context. Unsafe operations are deprecated and can be detected at compile-time, and will fail at run time. Non-deprecated operations will not fail at runtime.
- `dqm::harvesting::` restricts the interface to operations that are fully supported in harvesting. These are a superset of the operations allowed by `dqm::reco::`, but some functionality of `dqm::legacy::` remains deprecated.

Internally, there are a few more important classes:
- `MonitorElementData` encapsulates a ROOT Histogram (optional), the metadata for it (run, lumi, path), and a lock to protect the histogram from concurrent modifications.
- `MonitorElementCollection` is a dumb container of `MonitorElementData` objects used as a EDM product.

#### File formats

DQM data (mostly histograms, more specifically `MonitorElement`s) can be save in multiple different formats. The formats differ in how slow IO is.

- _DQMIO_: this is the "official" DQM data format. It is used for the DQMIO datatier stored after the RECO step in processing jobs. Histograms are stored in ROOT `TTree`s, which makes IO reasonably fast. EDM metadata is preserved, ME metadata is properly encoded and multiple runs/lumisections can be store in a single file. DQMIO data may also be preserved after _HARVESTING_ (to be iscussed).
- _Legacy ROOT_ (`DQM_*.root`): this is the most "popular" DQM data format. It is used for the uploads to DQMGUI and typically saved after the HARVESTING step. Histograms are stored in a `TDirectory`, which makes IO very slow but easy. Run/lumi metadata is not supported everywhere. We will continue to provide this format for compatibility with exisitng tooling, but might switch to producing it only on demand (to be discussed).
- _EDM event_: We might support writing `MonitorElementCollection`s directly into EDM event files, if a use case appears.

Data is archived for eternity by CMS computing in DQMIO format (TODO: confirm the precise rules), and in the DQMGUI index (hosted by CMSWEB, on disk, instant access to any ME). Legacy ROOT format files will remain available, but might be produced on-demand from one of the other formats. DQMIO is the recommended format for DQM data, including streaming to the DQMGUIs (not supported yet).

#### Processing environments

_See this section in the old system. No changes are expected._

### Modes of DQM operation

In the _RECO_ step, `DQMEDAnalyzer` plugins run and produce `MonitorElementCollection` products, which they put into the runs and lumi blocks after filling is finished. All handling of products is done in the base classes, the subsystem code only books and fills histograms. The `bookHistograms` callback remains as the only recommended way to book `MonitorElement`s, but since all operations are local to the `DQMStore`, this is not strictily required. To trigger all the `DQMStore` operations, the base class needs to receive many edm framework callbacks. If these callbacks are also needed in the plugin code, the respective `dqm*`-prefixed version should be implemented. Finally, DQMIO output module selects all `MonitorElementCollection` products and writes them into the DQMIO output file. The EDM framework and `DQMStore` ensure that only the minimal number of histograms required remains in memory.

To keep the amount of duplicate histograms in memory manageable, for the default `edm::stream::` based `DQMEDAnalyzer`, a single _master_ `DQMStore` holds all the histogram while the `DQMStore` instances local to the per-stream instances act as proxies to this master instance. The hand out `MonitorElement` objects backed by shared histograms held in the master `DQMStore`. The backing histograms are automatically switched as the stream enters new lumisections and runs, while bare pointers to the `MonitorElement` remain valid.

In the _HARVESTING_ `DQMHarvester` plugins run and read `MonitorElementCollection`s from the runs and lumisections. All MEs in the products are available to the plugin code by calling `get` on the `DQMStore` or `IGetter`. If a `MonitorElement` may be modified by the plugin code, it is cloned and the new instance will be put into the plugins `MonitorElementCollection` product. The collections produced by harvesters have a different label to prevent cyclical dependencies by default (explicit dependency declaration in the configuration can be used in ambiguous cases). Output modules collect *all* MEs and write them to output files (DQMIO or legacy ROOT format). Merging of partial run products is handled in the DQMIO input source.

Legacy plugins can keep using the `DQMStore` (their own instance) directly, and will not interact with other plugins in the job. They can still use the `DQMStore` to book and fill histograms, and have them saved into a file by calling `save` on the `DQMStore`. If interaction with other plugins is required, the necessary product handling can also be done manually.

### How do the compoments interact?

Each plugin has a pointer to its onw, local `DQMStore` instance. This object is created at intialisation time and availabel as a member variable or passed in from the base class in the form of `IBooker`/`IGetter` objects. Since the `DQMStore` contains mutable state (not only the histogram storage itself, but also e.g. the stateful booking interface with `cd`), interaction with the `DQMStore` is not thread safe, but since it is always local to the plugin, it can be safely used in `edm::one` and `edm::stream` modules. In `edm::global` modules, the accessing the `DQMStore` is not always safe, but that usual interactions in `bookHistograms` are fine.

To monitor a quantity, the plugin _books_ a `MonitorElement` with the `DQMStore`. This can happen at any time (legacy code) or only in a specific callback from the base class (`bookHistograms` in `DQMEDAnalyzer`), which is called once per run. The `MonitorElement` is identified by a path that should be unique. Multiple plugins can book histograms with the same path, but only one of them will be save in the end. The result of the booking call is a bare pointer to a `MonitorElement`.

The `MonitorElement` can now be filled with data using the `Fill` call, similar to a ROOT histogram. This can only happen *within* lumisections, since the `MonitorElement`s need to be prepared and saved in the lumi transitions -- in between lumisections, they may be invalid. `MonitorElement`s are backed by `MonitorElementData` objects, which contain ROOT TH1 objects. However, the TH1 object can't be accessed directly, to make sure all accesses do the required locking to make multi-threaded access safe. In case of `edm::stream` `DQMEDAnalyzer`s, there are individual `MonitorElement` instances per `edm::stream` instance of the plugin, but only one backing `MonitorElementData` object.

Whenever a histogram has finished filling (because the data that was supposed to go into it, e.g. a run or a lumisection, has been processed), the `MonitorElementData` project is moved into a `MonitorElementCollection` which is in turn moved into an EDM product. From now on, the histogram is immutable. Since the product is no longer available by the time the next lumisection/run starts, and the TH1 object is the only complete source of histogram properties from booking, the `MonitorElementData` might be cloned into a _prototype_ `MonitorElement` that exists temporarily in the `DQMStore` until the next lumisection/run starts. This _prototype_ can even be filled if there are fill calls during run/lumi transitions.

Other plugins can read the `MonitorElementCollection` products from runs/lumis and access them. For _HARVESTING_, the `DQMStore` automatically searches all known products when `MonitorElement`s are requested using the `get*` APIs. The `MonitorElementData` objects from the products are wrapped into `MonitorElement`s set to _read only_. Once a plugin attemps to modify such a _read only_ ME, the `MonitorElementData` is cloned, to prevent a modification of the histogram in the product. In _HARVESTING_, we also allow direct access to the underlying TH1 objects, but only after the histogram has been cloned and is owned by the plugin. These cloned histograms are in turn turned into new products, together with the freshly booked ones.

`MonitorElement`s typically collect statistics over a run or lumisection. This _scope_ of the `MonitorElement` can be set during booking. `DQMEDAnalyzer` by default save per-lumisection historgrams, except those base classes which can't. `DQMEDHarvester`s book per-lumi or per-run histograms depending on whether they are booked in lumi or run transitions.

`MonitorElement`s with a scope longer than a run, as they are need e.g. for multi-run harvesting, are planned but can't be fully supported, since EDM does not allow to have per-job products. Therefore, multi-run harvested histograms would need to be saved by the module that booked them.

For online monitoring, we might introduce an `edm::Service` that keeps references to *all* `MonitorElement`s in the system, to allow a live view even of per-run histograms.
