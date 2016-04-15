The SiPixelPhase1Common Framework
=================================

This framework is used for DQM on the upgraded Pixel Detector. Since it is designed to be geometry independent, it could also be used for the old detector, given the geometry-specific code paths and configuration are added. This framework goes deeper than what was used before, with the goal to move as much redundant work as possible from the DQM plugins into a central place.

This document decomposes into three sections:

   1. *The Concept* -- how the framework is intended to work.
   2. *How to actually use it* -- practical examples with explanation.
   3. *How it really works* -- explanations on the implementation.

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

This conceptual table consists of a left hand part, which contains inforamtion _about_ a datapoint, which decides _where the entry will go_. The right hand part has the actual value. The left hand side could have much more columns, basically every piece of information that we can collect about a datapoint should go there. On the right hand side, there also might be more than one quantitiy, however for simplicity we do not allow that in practice. We require the right hand side to be a 1D or 2D value, or a 0D value (which just means "is present"). To represent mulltiple quantities, we use multiple tables.

In this first table, every call to `fill`, which inserts an entry into a histogram, is one row. Such a table cannot be stored in practice, but we can still use it as a model for the specification language. The implementation will work around this problem later.

Now we want to derive histograms from this table. For this, we first recognize that every single row already has a histogram on its right hand side -- one with only one entry, but we can consider this datapoint a "atomic histogram". From these we now produce more interesting histograms by grouping rows of the table, and thier histograms together. For this, we demand that the tuples on the left side are unique. If the same tuple appears in multiple rows, we merge their histograms into one. For the sample table shown, and also in reality, most rows will still be unique. To get useful histograms, we now apply _projections_ (relational algebra term -- simpler than LA projections). A projection reduces the set of columns on the left hand side, thereby decreasing the number unique rows and forming histograms with more data.

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

As of now we discarded the information of the columns that we removed. However, we can also move colunms from the lefthand side to the right side and also remove colums on the right side if we are not interested in them. This will change the dimensionality of the histogram, which is fine (within technical limits -- `root` only allows a limited number of dimensions and we also have to keep track of which we want as x- and y-axis).

For now, we combined histograms by merging the data sets that the histogram is computed over (which is equivalent to summing up the bin counts). But there are also other ways to summarise histograms: For example, we can reduce a histogram into in single number (e.g. the mean), and then create a new plot (which may or may not be a histogram) of this derived quantitiy. One common special case of this are profiles, which show the mean of a set of histograms vs. some other quantity (they can also be considered a special representation of a 2D histogram, which is equivalent but we will ignore that for now). We can also create such histograms using projections. One way is to first reduce the right side histograms into a single number each and then move a column from the left side to the right side, using the value for the x-axis. Drawing the now 2D-values on the right side into a scatter plot might give us what we expect, but we cuold as well end up with a mess, since the x-axis values are not gurateed to be unique. They could also be unevenly distributed, which also does not give a nice plot.

Another way is to apply the projection, and concatenate the histograms that get merged along the x- or y-axis. Now it matters that we sorted the rows before, since the ordering of the rows will now dictate where the histograms go. If we reduced the histograms into single-bin histograms that show the extracted quantity (e.g. the mean) as the value of that bin, we will now end up with a nice profile. This is the preferred way to create profiles in the framework. The procedure outlined before is only used when it is necessary for technical reasons (DQM step1).

The framework provides a few simple commands to describe histograms using this model:

- The `groupBy` command projects onto the columns given as the first parameter, and combines the histograms by summing or concatenating along some axis, specified by the second parameter.
- The `reduce` command reduces the right-hand side histograms into histograms of a lower dimensionality. Typically, e.g. for the mean, this is a 0D-histogram, which means a single bin with the respective value. The quantitiy to extract is specified by the parameter.
- The `save`command tells the framework the we are happy with the histograms that are now in the table and that they should be saved to the output file. Proper names are inferred from the column names and steps specified before.

The framework also allows a `custom` command that passes the current table state to custom code, to compute things that the framework does not allow itself. However, this should be used carefully, since such code has to be haevily dependent on internal data structures.

Since the framework can follow see where quantities go during the computaion of derived histograms, it can also automatically keep track of the axis labels and ranges and set them to useful, consistent values automatically. For technical reasons the framework is not able to execute every conceivable specification, but it is able to handle all relevant cases in an efficient way. Since the specification is very abstract, it gievs the framework a lot of freedom to implement it in the most efficient way possible.

How to actually use it
----------------------

If you plugin derives from `SiPixelPhase1Base`, you should automatically have access to `HistogramManager`s, in the array `histo[...]`. Every HistogramManager shoould be used for one quantity. The indices to the array should be enum constants, use sth.like this

    enum {
      ADC, // digi ADC readouts
      NDIGIS, // number of digis per event and module
      MAP // digi hitmap per module
    };

in the C++ header to name your quantities and refer to them as `histo[ADC]`.

In the Analyzer code, call the `fill` function on the `histo[...]` whenever you have a datapoint to add. If you are just counting something (in the histograms above, all except the `ADC` are counts), use the `fill` function without any `double` argument. 

The `fill` function takes a number of mandatory and optional parameters the the left-hand side columns are derived from. You always have to pass a module ID (`DetId`), the `Event`is only for time-based things (if you want per lumisection or per bunchcrossing plots), and `row` and `col` are needed for ROC information, or if you want a 2D map of the module.

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

The second part initializes a default harvesting plugin, which is needed to exectute some specifications. 

The most important thing in the configuration are the specifications. Add them by setting `specs` in the `clone`. Since you can have many specifications per quantitiy, this has to be a `VPSet` as well. To make a specification use the `Specification()` builder:

    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping) 
                     .save()
    )

This will give you a default set of histogram, usually grouped by ladders etc. (but configurable). To add histograms per disk as well, add

    Specification().groupBy(DefaultHisto.defaultGrouping) 
                   .save()
                   .groupBy(parent(DefaultHisto.defaultGrouping)
                   .save()

You can also add histograms for all default partitions by adding `saveAll()`. 

To get profiles, add instead

    Specification().groupBy(DefaultHisto.defaultGrouping) # per-ladder and profiles
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),

If you want finer-grain histograms than per ladder, try

    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId") 
                   .save()

The `DetId` grouping in the end ensures module-level plots.

One of the more complicated things is counting things per event, as in e. g. the `NDigis` histograms. The specification for this is

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/Event") 
                     .reduce("COUNT") # per-event counting
                     .groupBy(DefaultHisto.defaultGrouping) 
                     .save()
                     
You can still add e.g. the profile snippet below to get profiles. For the per-event counting, you also need to call the per-event harvesting method at the end of your `analyze` method:
  histo[NDIGIS].executePerEventHarvesting(); 

For all the specifications, the steps up to the first `save` are executed in DQM step1. The histograms specified after the first `save` are created in DQM step2, Harvesting. Since step1 is very restricted due to multi-threading and performance requirements, not all specifications can be executed in step1. So, sometimes a `save` is necessary to make a specification work (this is e.g. the case with the profiles). After the first `save` however, you can or remove others freely to see or not see plots.


How it really works
-------------------

This section explains for every class in the framework what it does. This might also include hints how to use it correctly. Ordered roughly from simple to complex.

### AbstractHistogram

This is an unspectacluar wrapper aroud histogram-like things. It is used as the right-hand side of all tables that are actually kept in memory. It can contain either a root `TH1`, a `MonitorElement` (then the `th1` points to the MEs TH1) or a simple counter (`int`). The counter can be used independently of the histogram and is used to concatenate histograms in harvesting and count things per event in step1.

### SiPixelPhase1Base

This is the base class that all plugins should derive from. It instantiates `HistogramManager`s from config, sets up a `GeometryInterface`, and callls the `book` method of the `HistogramManager`s. If you need anything special, you can also do this yourself ad ignore the base class.

The same file delcares the `SiPixelPhase1Harvester`, which is very similar to the base class but is an `DQMEDHarvester` and calls the harvesting methods. Usually this does not need to be modified, it is enough instantiate a copy and set the configuration to be the same as for the analyzer. If you want too use a `custom` step, derive from this class and call the `setCustomHandler` on the HistogramManager before the actual harvesting starts.

### SummationSpecification

A dumb datastructure that represents a specifiaction, as introduced above. The specification is in an inernal, slightly different format, which is created by the SpecfiactionBuilder in Python code; the C++ SummationSpecification just takes the `PSet` structure and does no special processing, except for converting columns from the string form to the efficient `GeometryInterface` form.

### SpecficationBuilder

This is the Python class that provides the specification syntax used above. It creates a nested structure of `PSet`s with all the information, and also transforms the specification from the simple language explaind above into something closer to the implementation. This includes 
  - attaching the DQM step (here called `Stage`) where the step is to be executed to every `Step`.
  - creating different internal commands for `groupBy` steps, depending on whether it is a summation or a extension. A detail here is that while a normal `GROUPBY`has a list of columns to keep, the internal `EXTEND_*` lists the columns to be dropped, to simplify step1 processing.
  - special-casing "/Event" grouping, which has to be done using a special `STAGE1_2` only used for this purpose.
  - checking for all sorts of well-formedness, to catch possible problems as soon as possible.

### GeometryInterface

This is a class, but one of the more important things are the types declared within it. They get used extensively in the HistogramManager, but we try to avoid that they leak into the plugin code. You need to know these types if you want to touch the actual implemenation of the framework.

#### GeometryInterface::Value

The value of "cells" in the table. Defined to be `int`.

#### GeometryInterface::ID

The internal form of a column name, like `"PXLadder"` or `"DetId"`. Also defined to be `int`. To get an ID, call `intern` with the name. To get back to the string form, use `unintern`. The ID to string mapping is done with a `std::map` that grows automatically, so there is no such thing as a list of reserved names. 

Only pass things returned by `intern` as an `ID`. Other values can make things crash.

(to be continued)
