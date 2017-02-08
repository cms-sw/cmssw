The SiPixelPhase1Common Framework
=================================

This framework is used for DQM on the upgraded Pixel Detector. Since it is designed to be geometry independent, it can also be used for the old detector, up to some geometry-specific code paths. This framework goes deeper than what was used before, with the goal to move as much redundant work as possible from the DQM plugins into a central place.

This document decomposes into three sections:

   1. *The Concept* -- how the framework is intended to work.
   2. *How to actually use it* -- practical examples with explanation.
   3. *How it really works* -- explanations on the implementation.

Further information can be found on the TWiki page: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PixelDQMPhaseI . Note that parts migh be historical and not reflect the current state.

The Concept
-----------

The central thing in DQM are histograms. Creating histograms, and summing up their content in other plots, is what most of the DQM code does. To make this as easy as possible, the framework provides the `HistogramManager`, which allows managing large numbers of related histograms and computing derived plots from them in a highly configurable way. Only two tasks remain for the DQM plugin code: First, collect the data to go into the histograms and call `fill`. Second, configure the HistogramManager with a specification of the histograms to output. 

To specify what the output histograms should look like and where they should go, a simple, yet expressive model is used, along with a simple configuration language to write down the specifications. The model is based on a large table and a subset of relational algebra to modify that table.

| Layer  | Ladder | Module | row | col | I | adc readout|
|--------|--------|--------|-----|-----|---|------------|
|      1 |      3 |      2 |  45 |  56 | I |         23 |
|      1 |      4 |      1 |  12 | 154 | I |        120 |
|      2 |      1 |      2 |  36 |  37 | I |          5 |
|      1 |      3 |      2 |  45 |  29 | I |         13 |
|      3 |      4 |      3 |  13 |  40 | I |         29 |
|      3 |      3 |      2 |  83 |  47 | I |        234 |
|      1 |      1 |      3 |  38 |  92 | I |         34 |
|      1 |      3 |      1 |  55 |  73 | I |         57 |

This conceptual table consists of a left hand part, which contains information _about_ a datapoint, which decides _where the entry will go_. The right hand part has the actual value. The left hand side could have much more columns, basically every piece of information that we can collect about a datapoint should go there. On the right hand side, there also might be more than one quantity, however for simplicity we do not allow that in practice. We require the right hand side to be a 1D or 2D value, or a 0D value (which just means "is present"). To represent multiple quantities, we use multiple tables.

In this first table, every call to `fill`, which inserts an entry into a histogram, is one row. Such a table cannot be stored in practice, but we can still use it as a model for the specification language. The implementation will work around this problem later.

Now we want to derive histograms from this table. For this, we first recognize that every single row already has a histogram on its right hand side -- one with only one entry, but we can consider this datapoint a "atomic histogram". From these we now produce more interesting histograms by grouping rows of the table, and their histograms together. For this, we demand that the tuples on the left side are unique. If the same tuple appears in multiple rows, we merge their histograms into one. For the sample table shown, and also in reality, most rows will still be unique. To get useful histograms, we now apply _projections_ (relational algebra term -- simpler than LA projections). A projection reduces the set of columns on the left hand side, thereby decreasing the number unique rows and forming histograms with more data.

_Project on Layer, Ladder, Module_

| Layer  | Ladder | Module | I | adc readout|
|--------|--------|--------|---|------------|
|      1 |      1 |      3 | I |         34 |
|      1 |      3 |      2 | I |   [23, 13] |
|      1 |      3 |      1 | I |         57 |
|      1 |      4 |      1 | I |        120 |
|      2 |      1 |      2 | I |          5 |
|      3 |      3 |      2 | I |        234 |
|      3 |      4 |      3 | I |         29 |

_Project on Layer, Ladder_

| Layer  | Ladder | I | adc readout|
|--------|--------|---|------------|
|      1 |      1 | I |         34 |
|      1 |      3 | I |[23, 13, 57]|
|      1 |      4 | I |        120 |
|      2 |      1 | I |          5 |
|      3 |      3 | I |        234 |
|      3 |      4 | I |         29 |


_Project on Module_

| Module | I | adc readout|
|--------|---|------------|
|      1 | I |   [57, 120]|
|      2 | I |   [23, 13, 5, 234]|
|      3 | I |   [29, 34]|

Note how we also sorted the rows. The order of the rows does not matter for now, but it will, later.

As of now we discarded the information of the columns that we removed. However, we can also move columns from the left hand side to the right side and also remove columns on the right side if we are not interested in them. This will change the dimensionality of the histogram, which is fine (within technical limits -- `root` only allows a limited number of dimensions and we also have to keep track of which we want as x- and y-axis).

For now, we combined histograms by merging the data sets that the histogram is computed over (which is equivalent to summing up the bin counts). But there are also other ways to summarise histograms: For example, we can reduce a histogram into in single number (e.g. the mean), and then create a new plot (which may or may not be a histogram) of this derived quantity. One common special case of this are profiles, which show the mean of a set of histograms vs. some other quantity (they can also be considered a special representation of a 2D histogram, which is equivalent but we will ignore that for now). We can also create such histograms using projections. One way is to first reduce the right side histograms into a single number each and then move a column from the left side to the right side, using the value for the x-axis. Drawing the now 2D-values on the right side into a scatter plot might give us what we expect, but we could as well end up with a mess, since the x-axis values are not guaranteed to be unique. They could also be unevenly distributed, which also does not give a nice plot.

Another way is to apply the projection, and concatenate the histograms that get merged along the x- or y-axis. Now it matters that we sorted the rows before, since the ordering of the rows will now dictate where the histograms go. If we reduced the histograms into single-bin histograms that show the extracted quantity (e.g. the mean) as the value of that bin, we will now end up with a nice profile. This is the preferred way to create 1D summary profiles in the framework. The procedure outlined before is also used, e.g. for 2D profiles. It has the advantage that t can be computed in DQM step1, while the second method reqires intermediate historams saved for step2 (harvesting).

The framework provides a few simple commands to describe histograms using this model:

- The `groupBy` command projects onto the columns given as the first parameter, and combines the histograms by summing or concatenating along some axis, specified by the second parameter.
- The `reduce` command reduces the right-hand side histograms into histograms of a lower dimensionality. Typically, e.g. for the mean, this is a 0D-histogram, which means a single bin with the respective value. The quantity to extract is specified by the parameter.
- The `save`command tells the framework that we are happy with the histograms that are now in the table and that they should be saved to the output file. Proper names are inferred from the column names and steps specified before.

The framework also allows a `custom` command that passes the current table state to custom code, to compute things that the framework does not allow itself. However, this should be used carefully, since such code has to be heavily dependent on internal data structures.

Since the framework can see where quantities go during the computation of derived histograms, it can also automatically keep track of the axis labels and ranges and set them to useful, consistent values automatically. For technical reasons the framework is not able to execute every conceivable specification, but it is able to handle all relevant cases in an efficient way. Since the specification is very abstract, it gives the framework a lot of freedom to implement it in the most efficient way possible.


How to actually use it
----------------------

If your plugin derives from `SiPixelPhase1Base`, you should automatically have access to `HistogramManager`s, in the array `histo[...]`. Every HistogramManager should be used for one quantity. The indices to the array should be enum constants, use something like this

    enum {
      ADC, // digi ADC readouts
      NDIGIS, // number of digis per event and module
      MAP // digi hitmap per module
    };

in the C++ header to name your quantities and refer to them as `histo[ADC]`.

In the Analyzer code, call the `fill` function on the `histo[...]` whenever you have a datapoint to add. If you are just counting something (in the histograms above, all except the `ADC` are counts), use the `fill` function without any `double` argument. 

The `fill` function takes a number of mandatory and optional parameters that the left-hand side columns are derived from. You always have to pass a module ID (`DetId`), the `Event`is only for time-based things (if you want per lumisection or per bunchcrossing plots), and `row` and `col` are needed for ROC information, or if you want a 2D map of the module.

Now, you need to add some specifications in the config file.

    SiPixelPhase1DigisADC = DefaultHisto.clone(
      ...
    )
    SiPixelPhase1DigisNDigis = DefaultHisto.clone(
      ...
    )
    SiPixelPhase1DigisHitmaps = DefaultHisto.clone(
      ...
    )
   
    SiPixelPhase1DigisConf = cms.VPSet(
      SiPixelPhase1DigisADC,
      SiPixelPhase1Digis,
      SiPixelPhase1DigisHitmaps 
    )

Add a `VPSet` with one clone of a `DefaultHisto` per histogram you want to use. These have to be the same number and same order as in the `enum` in the C++ code. In the `clone` you can set various parameters, refer to `HistogramManager_cfi.py` for details.

The `VPSet` alone will not do anything. Add it to the configuration of your plugin:

    SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1Digis",
            src = cms.InputTag("simSiPixelDigis"), 
            histograms = SiPixelPhase1DigisConf,
            geometry = SiPixelPhase1Geometry
    )
    SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
            histograms = SiPixelPhase1DigisConf,
            geometry = SiPixelPhase1Geometry
    )

The second part initializes a default harvesting plugin, which is needed to execute some specifications. 

The most important thing in the configuration are the specifications. Add them by setting `specs` in the `clone`. Since you can have many specifications per quantity, this has to be a `VPSet` as well. To make a specification use the `Specification()` builder (note that the `VPSet` used here is defined in `HistogramManager_cfi.py`, it akes a `cms.VPSet` but gives you a bit more freedom with the arguments):

    specs = VPSet(
      Specification().groupBy("PXBarrel/PXLayer/PXLadder") 
                     .save(),
      Specification().groupBy("PXForward/PXDisk/PXBlade") 
                     .save(),
    )

This will give you a set of histograms, grouped by ladders and blades. The `groupBy` line specifies 3 levels, each for FPIX and BPIX independently. In the output, this will lead to nested folders of the given names, where the last level of folders (one per ladder/blade) contains the histograms. The string lists columns separated by `/`, which has the same meaning as in a directory hierarchy. For the histograms that are created, the order of columns does not matter; however, the order defines the directory nesting and where the histograms go. (Also, if you use the shorthand `saveAll`, you get summations defined by the ordering/directory hierarchy). Make sure you always list columns in the same order within one specification, otherwise you might hit unsupported or buggy cases. Usually you will need two specifications each, one for barrel and one for forward. There is no central list of available column names, as new extractors could be added anywhere. However, most are defined in the `GeometryInterface`, which will also output a list of known columns at runtime (if sufficient debug  logging is enabled). 

To add histograms per disk as well, add

    Specification().groupBy("PXBarrel/PXLayer/PXLadder") 
                   .save()
                   .groupBy("PXBarrel/PXLayer")
                   .save(),
    Specification().groupBy("PXForward/PXDisk/PXBlade") 
                   .save()
                   .groupBy("PXForward/PXDisk")
                   .save(),

You can also add histograms for all mentioned levels by adding `saveAll()`. 

To get a summary profle, add instead

    Specification().groupBy("PXBarrel/PXLayer/PXLadder") # per-ladder and profiles
                   .save()
                   .reduce("MEAN")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_X")
                   .saveAll(),
    Specification().groupBy("PXForward/PXDisk/PXBlade") # per-ladder and profiles
                   .save()
                   .reduce("MEAN")
                   .groupBy("PXForward/PXDisk", "EXTEND_X")
                   .saveAll(),
                   
For quantities where it makes sense to have per-module plots, you can add a specification like this:

    Specification(PerModule).groupBy("PXForward/PXDisk/DetId").save(),
    Specification(PerModule).groupBy("PXBarrel/PXLayer/DetId").save(),

The `PerModule` parameter (defined in `HistogramManager_cfi.py`, allows the per-module histograms to be turned off by default.

One of the more complicated things is counting things per event, as in e. g. the `NDigis` histograms. The specification for this is something like this:

      Specification().groupBy("PXBarrel/PXLayer/PXLadder/DetId/Event") 
                     .reduce("COUNT") # per-event counting
                     .groupBy("PXBarrel/PXLayer/PXLadder") 
                     .save(),
      Specification().groupBy("PXForward/PXDisk/PXBlade/DetId/Event") 
                     .reduce("COUNT") # per-event counting
                     .groupBy("PXForward/PXDisk/PXBlade") 
                     .save(),
                     
We group by module and Event first, since this is the range that we want counted, then we reduce and group as usual to get a histogram of the counts. You can still add e.g. the profile snippet below to get profiles. For the per-event counting, you also need to call the per-event harvesting method at the end of your `analyze` method:
    histo[NDIGIS].executePerEventHarvesting(); 

However, usually you don't have to write specifications yourself. There are a set of predefined specifications (defined in `HistogramManager_cfi.py`) that can be used for most standard histograms:

    StandardSpecifications1D,
    StandardSpecificationsTrend,
    StandardSpecifications2DProfile,
    StandardSpecifications1D_Num,
    StandardSpecificationsTrend_Num,
    StandardSpecifications2DProfile_Num,

The `_Num` versions perform a NDigis-like counting, while the others expect a simple 1D quantity. They produce, respectively, 1D histograms and summary profiles, a trend plot over time and a profile of the quantity as a 2D map of each layer.

For more complicated cases, first check whether something similar already exists. In that case, you might be able to steal a working specification there...

For all the specifications, the steps up to the first `save` are executed in DQM step1. The histograms specified after the first `save` are created in DQM step2, Harvesting. The types of specifications that can be executed in step1 and step2 are fairly different, and also the semantics change a bit (s explained above, with to different approaches for EXTEND). In general, step2 does not allow any 2D plots at the moment.


How it really works
-------------------

This section explains for every class in the framework what it does. This might also include hints how to use it correctly. Ordered roughly from simple to complex.

### AbstractHistogram

This is an unspectacular wrapper around histogram-like things. It is used as the right-hand side of all tables that are actually kept in memory. It can contain either a root `TH1`, a `MonitorElement` (then the `th1` points to the MEs TH1) or a simple counter (`int`). The counter can be used independently of the histogram and is used to concatenate histograms in harvesting and count things per event in step1.

### SiPixelPhase1Base

This is the base class that all plugins should derive from. It instantiates `HistogramManager`s from config, sets up a `GeometryInterface`, and calls the `book` method of the `HistogramManager`s. If you need anything special, you can also do this yourself ad ignore the base class.

The same file declares the `SiPixelPhase1Harvester`, which is very similar to the base class but is an `DQMEDHarvester` and calls the harvesting methods. Usually this does not need to be modified, it is enough instantiate a copy and set the configuration to be the same as for the analyzer. If you want too use a `custom` step, derive from this class and call the `setCustomHandler` on the HistogramManager before the actual harvesting starts.

### SummationSpecification

A dumb datastructure that represents a specification, as introduced above. The specification is in an internal, slightly different format, which is created by the SpecificationBuilder in Python code; the C++ SummationSpecification just takes the `PSet` structure and does no special processing, except for converting columns from the string form to the efficient `GeometryInterface` form.

Note that the format of the Specification in there is fairly rigid and the `HistogramManager` assumes a lot of things there. The explanation is in comments in the `SpecificationBuilder`.

### SpecficationBuilder

This is the Python class that provides the specification syntax used above. It creates a nested structure of `PSet`s with all the information, and also transforms the specification from the simple language explained above into something closer to the implementation. This includes 
  - attaching the DQM step (here called `Stage`) where the step is to be executed to every `Step`.
  - creating different internal commands for `groupBy` steps, depending on whether it is a summation or a extension. A detail here is that while a normal `GROUPBY`has a list of columns to keep, the internal `EXTEND_*` lists the columns to be dropped, to simplify step1 processing.
  - special-casing "/Event" grouping, which has to be done using a second `FIRST` step of type `COUNT`.
  - checking for all sorts of well-formedness, to catch possible problems as soon as possible.

### GeometryInterface

This is a class, but one of the more important things are the types declared within it. They get used extensively in the HistogramManager, but we try to avoid that they leak into the plugin code. You need to know these types if you want to touch the actual implementation of the framework.

#### GeometryInterface::InterestingQuantities

This is a dumb struct that holds a `DetId` and other useful stuff about a datapoint, namely a pointer to the event information and row/column. Note that a bare pointer is used here as a _maybe_: it could be `null` on any occasion, always check for that.

#### GeometryInterface::Value

The value of "cells" in the table. Defined to be `int`. The reserved value `UNDEFINED` signals that this value is missing.

#### GeometryInterface::ID

The internal form of a column name, like `"PXLadder"` or `"DetId"`. Also defined to be `int`. To get an ID, call `intern` with the name. To get back to the string form, use `unintern`. The ID to string mapping is done with a `std::map` that grows automatically, so there is no such thing as a list of reserved names. 

Only pass things returned by `intern` as an `ID`. Other values can make things crash.

#### GeometryInterface::Column

This is what is actually used to represent a column in the code most of the time. Unlike the ID, this can represent something like `PXLadder|PXBlade`. The implementation is a `std::array<ID, 2>` (keep in mind that this thing needs to be explicitly initialized!). This means at the moment only two names are allowed using the `|` (more is possible, but at a significant performance and slight complexity cost). There is some magic around the `Column` type to make things work smoothly: By default, we expect `Column`s to be _normalized_, that is the second ID is 0. Then we can just use the `col[0]` as the ID of the column. The `GeometryInterface` will normalize `Column`s once a value is assigned and we know which value was actually used. Fuzzy matching is used to allow referring to multi-ID columns with a normalized `Column` and vice versa. Most code should not have to care about it, it can just pass around `Column` values and use `pretty` to create a string representation.

#### GeometryInterface::Values

This is a mapping from `Column` names to `Value` values. While a `std::map` could be used perfectly, for performance reasons a different implementation was used. This is a `std::vector<std::pair<Column, Value>>`. It turns out using map operations cen be avaoided in practice, by re-extracting things from the `InterestingQuantities` in case of doubt. The methods providing and accepting `Column`-`Value`-pairs should be preferred, in case we want to add or-columns back in. `Values` are used extensively as keys for `std::map`, but there are some caveats: The ordering in the vector (insertion order) matters (this is why it is usually a good idea to fill a `Values` object by looping over the `columns` of the first specified grouping). 

In terms of memory management, it can be a good idea to keep one `Values` object around and `erase`, assign, and `swap` the vector inside it directly. This allows handling `Values` without any heap allocations (note that a `map` allocates all nodes on the heap, and a vector allocates its backing array on the heap, even if they are stack values in the code). This is the main point for having the `Values` type.

#### GeometryInterface itself

The heart of the GeometryInterface, apart from the types, is a mapping from `ID`s to extractor functions (stored as `std::function<Value(InterestingQuantities const&)>`), that can be used to obtain column values from `DetId`s. The mapping is a `std::vector` indexed by `ID`s (this is the reason for having a dense set of IDs in the first place). Since this mapping is a bit fragile, only the provided accessor functions should be used. Extractors are added while the `GeometryInterface` is _loaded_, which happens independent from construction. Use `addExtractor(intern("Name"), <lambda>, min, max)` to add more. The range passed to `addExtractor` cannot in general be precise, but it is used for histogram ranges and EXTEND in step1.

### HistogramManager

This is where most of the implementation hides. 

#### HistogramManager::Table

This is the most important datastructure in the `HistogramManager`. It is a `std::map<Values, AbstractHistogram>`. This represents the conceptual table outlined above and is used in harvesting to perform all operations. In step1, things are a bit more complicated but the `Table` is still the main structure. Typically the `AbstractHistogram`s in a table always have a `th1` assigned (may or may not be a ME), unless it is a step1 counter. However, it can happen that you hit a row that was never touched before and does not have a `th1` (e. g. if a foreign `DetId` was passed in), and these should be ignored. The histogram concatenation code relies on the lexicographical order of the map, which `std::map` does provide. The fast path for `fill` should make no more than one lookup in this table (per spec). 

The `HistogramManager` holds two sets of Tables, one for counters, and another one for histograms, but they are of the same type. Also harvesting uses the same type, even though no code is shared between harvesting and step1.

#### Booking Process

From the DQM booking function, `HistogramManager::book` is called. This method takes the list of all known modules from the `GeometryInterface` and executes the step1 commands for each spec. It tracks metadata like ranges and labels during this and sets them in the end, when the MEs are booked for each new table entry.

#### Filling 

The fill function asks the `GeometryInterface` for the column values for the module data passed in, and executes the step1 commands on it. This is much easier than in the booking before, but also has to be much faster. A special case here is per-event counting, where the fill function does not fill a histogram but increments a counter (in an AbstractHistogram). Later, when `executePerEventHarvesting` is called, the counts are ran though the remaining step1 commands and fill is actually called, using the same code as the actual `fill`.

#### Harvesting

For the harvesting, first the internal table structure has to be repopulated (remember, we are in a completely new process now). To do this, `loadFromDQMStore` mirrors the booking process (massively stripped down and not actually sharing code) and `get`s the MEs from the `DQMStore`. So, in sum there are 3 places where step1 commands are interpreted: in booking, in filling, and in loading. Make sure to keep them in sync when anything changes. 

Afterwards, a classical interpreter loop runs over the step2 commands and calls methods to actually execute them, by updating the table structure. This is mostly straight-forward code. Booking happens as the execution progresses, and all metadata is tracked in the `TH1` fields (labels, range).

In the `custom` command, a custom handler function (that has to be set beforehand) is called. This is again a lambda (using inheritance would be possible as well, but it tends to be a mess in C++), which gets passed the internal table state. It is explicitly allowed to just save a copy of that state for later use, which may be necessary to compute quantities that depend on multiple other quantities (e.g. efficiencies). Later, a `custom` step on a different HistogramManager (possibly one that did not record any data in step1) can take the saved tables, create derived quantities and put them into the histograms in its own table, which the HistogramManager already booked with consistent names (also doing further summation, if specified).

Alternativey, the `custom` handler gets passed the `IBooker` and `IGetter` objects from the `DQMStore`, so things which cannot be booked using the framework can be handled manually. But then you also need to handle folder- and object names your self; best is to use the path names of the MEs you find in the table and derive new ME names from the existing ME names.
