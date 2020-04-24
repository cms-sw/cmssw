## Plugins
### APVCyclePhaseDebuggerFromL1TS
It is an `EDAnalyzer` which can be used to monitor the content of the `Level1TriggerScalersCollection` produced by SCAL. Histograms are filled for each event with the orbit number of the last: resynch, hard reset, EC0, OC0, Test Enable and Start signals. Therefore the histograms show spikes at the orbit number values when the signals have been issued with the a number of entries equal to the number of events between two signals. In addition two histograms are filled with the differences of the orbit numbers between EC0 and resync and resync and hard reset every time a new resync or a new hard reset has been issued. The configuration requires simply the name of the product with the `Level1TriggerScalersCollection` and, usually, it is `scalersRawToDigi`, as defined in the default configuration.

### APVCyclePhaseProducerFromL1TS
It is an `EDProducer` which produces an `APVCyclePhaseCollection` which contains, for each Tracker partition, an offset, between 0 and 69, to be used to compute the position of each event in the Tracker APV readout cycle from the orbit number and the bunch crossing number using `orbit*3564+bx-offset`. More details can be found [in this page](https://twiki.cern.ch/twiki/bin/view/CMS/APVReadoutCycle). The standard way of using it is to let the EventSetup provide the parameters using the record `SiStripConfObject` with the label `apvphaseoffsets` (configurable). If the configuration parameter `ignoreDB` is set to true, then the parameters are read from the configuration file and are: `defaultPartitionNames` usually equal to `TI,TO,TP,TM`, `defaultPhases` which contains a vector (one value per partition) of offsets which are used as offset when no resync has been issued, `magicOffset` which is used to correct the orbit number of the last resync to compute the phase when a resync has been issued using the expression: `(defaultPhase + (lastresyncorbit+magicOffset)*3564%70)%70`, and `useEC0` if we want to use `lastEC0orbit` instead of `lastresyncorbit` to compute the phase. Examples of configurations can be found in the `python` directory.

### ...BigEventsDebugger
There are three specializations of the `EDAnalyzer` `plugins/BigEventsDebugger` template: `ClusterBigEventsDebugger`, `DigiBigEventsDebugger` and `RawDigiBigEventsDebugger`. This plugin produces histograms of the distribution of the digis in SiStrip clusters, or SiStrip digis, or SiStrip raw digis found in configurable subsets of modules. By default (`want1dHisto` is `True` by default) only the histogram of the digi position is produced. The histograms of the average ADC counts vs the strip number can be activated with the configuration boolean `wantProfile` and the 2D histograms of the digi ADC counts vs the strip number can be activated with the configuration boolean `want2dHisto`. If the boolean `foldedStrips` is `True`, all the strips are folded in the range 0-255. If the boolean `singleEvents` is `True` one set of histograms per event is created in order to take a snapshot of single events. The subsets of modules whose digi distributions have to be monitored are selected with the configuration parameter `selections` which has to be provided with a vector of PSet with the parameter `selection` to configure `DetIdSelector` objects and the parameter `label` with a string used in the histogram titles. Example of configurations can be found in `python/clusterbigeventsdebugger_cfi.py`, `python/digibigeventsdebugger_cfi.py` and `python/rawdigibigeventsdebugger_cfi.py`. An Example of how to use this `EDAnalyzer` can be found in the configuration file `test/TIDTECInnerRingInvestigator_cfg.py`

### CommonModeAnalyzer
It is an `EDAnalyzer` that produces histograms of the common mode values provided by the FEDs and by the Strip unpacker. The user can define one or more subset of modules whose common mode values have to be analyzed. The definition of these subsets is done with the configuration parameter `selections` and the syntax used for the `DetIdSelector` class has to be used for the parameter `selection` while the parameter `label` is used to define the labels used in the histogram names and titles associated to that selection. The boolean parameters `ignoreFEDBadMod` and `ignoreNotConnected` can be used to configure the analyzer to ignore the modules which are declared as bad by the unpacker and the channels which are not connected, respectively: setting both parameters as `True` removes all the common mode values at zero and it help to identify genunely low common mode values. The parameter `digiCollection` is used to define the name of the common mode collection and the parameter `badModuleDigiCollection` is used to define the name of the collection of bad modules produced by the unpacker. The parameter `historyProduct` defines the name of the collection of `EventWithHistory` objects, the parameter `apvPhaseCollection` defines the collection of APV phases offsets and the parameter `phasePartition` defines the partition whose APV phase has to be considered: using `All` means that the phases of the four partitions have to agree. These parameters are needed because one of the histogram which is produced is the average common mode value as a function of the distance of the events from the start of the APV cycle after the previous L1A and, therefore, the distance from the previous L1A is needed, using `EventWithHistory` and the APV cycle phase is needed. The other histograms which are produced are: the distribution of the common mode values in all the events and in all the channels belonging to one of the selections, the average common mode value as a function of the orbit number (time evolution), the average common mode values as a function of the BX number, the distribution of the number of modules and APVs whose common mode values have been monitored per event.
An example of configuration can be found in `python/commonmodeanalyzer_cfi.py`. The file `test/commonmodeanalyzer_cfg.py` is an example of a full configuration.

### ...MultiplicityProducer
There are three specializations of the `EDProducer` `plugins/MultiplicityProducer.cc` template: `SiStripMultiplicityProducer`, `SiPixelMultiplicityProducer`, `SiStripDigiMultiplicityProducer`. This plugin produces simple objects which contain the numbers of strip/pixel clusters or strip digis in subsets of the detector in a given event. 
Examples of configurations can be found in:
* `python/sipixelclustermultiplicityprod_cfi.py`
* `python/sistripclustermultiplicityprod_cfi.py`
* `python/sistripdigimultiplicityprod_cfi.py`

The configuration must contain the name of the collection and the parameter `wantedSubDets` which define the subsets of the detector whose multiplicity has to be computed. Each element of this parameter is a `PSet` which contains an integer index, `detSelection`, used to identify the subset by the `EDAnalyzer`s which will use this product, a string, `detLabel`, for the moment dummy, and a vector of strings, `selection` which contains the configuration for a `DetIdSelector` object which define which part of the detector has to be considered. This last parameter can be omitted if the index `detSelection` is between 0 and 6. In that case "0" means the full tracker and "1-6" correspond to the different subdetector according to the detid schema. The optional parameter `withClusterSize` has to be set `True` if instead of counting the number of clusters we want to count the number of digis which compose the clusters. 

###MultiplicityInvestigator
It produces histograms related to the cluster or digi multiplicities using the products of the `MultiplicityProducer` plugins. In the configuration the name of the multiplicity product has to be provided as well as the list of multiplicity values to be used. This information is provided by the parameter `wantedSubDets` which contains `PSet` with the parameter `detSelection` which has to match the one specified in the `MultiplicityProducer`, a `detLabel` which will be used to define the histograms names and titles, and `binMax` which is used to define the range of the histograms. Boolean configuration parameters are used to define the set of histograms:
* if `wantInvestHist` is true the multiplicity distribution and the average multiplicity vs the orbit number is produced. Examples of configurations are `python/spclusmultinvestigator_cfi.py`, `python/ssclusmultinvestigator_cfi.py` and `python/ssdigimultinvestigator_cfi.py`.
* if `wantVtxCorrHist` is true the correlations of the multiplcities with the vertex multiplicity are produced. The configuration must contains also the name of the vertex collection and a `wantedSubDets` parameter has to be included in the `digiVtxCorrConfig` `PSet`. Examples of configurations are: `python/spclusmultinvestigatorwithvtx_cfi.py`, `python/ssclusmultinvestigatorwithvtx_cfi.py`, `python/ssdigimultinvestigatorwithvtx_cfi.py`
* if `wantLumiCorrHist` is true the correlation of the multiplicities with the instantaneous BX luminosities are produces. The name of the lumi producer has to be provided as well as a `wantedSubDets` parameter in the `PSet` `digiLumiCorrConfig`. Examples of configurations are `python/spclusmultlumicorr_cfi.py`, `python/ssclusmultlumicorr_cfi.py` and `python/ssdigimultlumicorr_cfi.py`.
* if `wantPileupCorrHist` is true the correlation of the multiplcities with the pileup (MC only) are produced. The configuration must contain the name of the pileup summary collection and a `wantedSubDets` parameter in the `PSet` `digiPileupCorrConfig`. Examples of configurations are `python/spclusmultpileupcorr_cfi.py`, `python/ssclusmultpileupcorr_cfi.py` and `python/ssdigimultpileupcorr_cfi.py`
* if `wantVtxPosCorrHist` is true the correlation of the multiplcities with the (MC) main vertex z position are produced. The configuration ust contain the name of the mc vertex collection and a `wantedSubDets` parameter in the `PSet` `digiVtxPosCorrConfig`. Examples of configurations are: `python/spclusmultvtxposcorr_cfi.py`, `python/ssclusmultvtxposcorr_cfi.py` and `python/ssdigimultvtxposcorr_cfi.py`. 

### MultiplicityCorrelator
It correlates two multiplicity values from different part of the detectors. The configuration contains a `VPSet`, `correlationConfigurations`, which contains a `PSet` for each correlation where the names of the multiplicity maps `xMultiplicityMap` and `yMultiplicityMap`, the selection indices, `xDetSelection` and `yDetSelection`, the labels, `xDetLabel` and `yDetLabel`, the number of bins and the max value of the histogram range are provided. In addition booleans are available to activate the most memory-consuming plots. An example of configuration is `python/multiplicitycorr_cfi.py`.

### MultiplicityTimeCorrelations
It produces histograms to correlate the multiplicities obtained from `MultiplicityProducer` with several parameters like the BX number, the position of the event in the APV cycle, the distance from the previous L1A and combinations of these quantities. It requires a `wantedSubDets` `VPSet` and the name of the multiplicity map collection to define the multiplicity values to be considered, the name of the `EventWithHistory` product and the name of the `APVPhaseCollection` product. Additional parameters are available to define the range and binning of the histograms and to preselect the events to be used (obsolete options now that there are dedicated `EDFilter`s in this package). Examples of configurations are `python/spclusmulttimecorrelations_cfi.py`, `python/ssclusmulttimecorrelations_cfi.py` and `python/ssdigimulttimecorrelations_cfi.py`. 

### OccupancyPlots
This `EDAnalyzer` requires as input the products of the `...MultiplicityProducer` `EDProducers` described above. Two vectors of input tags have to be provided: `multiplicityMaps`, which is expected to contain the cluster multiplicities as determined by the `...MultiplicityProducer` with the option `withClusterSize=False`, and `occupancyMaps`, which is expected to contain the digi in cluster multiplicities as determined by the `...MultiplicityProducer` with the option `withClusterSize=True`. This plugin produces two histograms with the average number of clusters or digis in clusters in each module subsets defined in the configuration of the `...MultiplicityProducer` modules whose products are used as input: each bin corresponds to one of those subsets and the bin number correspond to the parameter `detSelection` in the module subset definition PSet (look at `python/occupancyplotsselections_simplified_cff.py` as an example). The parameter `wantedSubDets` has to be provided to this plugin and has to be consistent with the equivalent module selections provided to the `...MultiplicityProducer` modules: these module selections are used to fill additional histograms with the average module subset positions (radius, z, x, y, ...) and the total number of channels (pixels or strips) in each module subsets.

## Configurations
More details can be found in the README file in the `test` directory
* `test/OOTmultiplicity_cfg.py` to study the multiplicity of pixel and strip clusters in filled and empty bunch crossings using zero bias and random triggers. It requires the macros described below to show its results.
* `test/apvphaseproducertest_cfg.py` to study the correlation of the strip multiplicities with the APV cycle and determine if the knowledge of the APV cycle phase offsets is still under control
* `test/bsvsbpix_cfg.py`to correlate the beamspot position with the average position of the BPIX ladders. It requires the macros described below to show its results.
* `test/OccupancyPlotsTest_cfg.py` to measure the hit occupancy and multiplicity in the different detector region (rZ view). It requires the macros described below to show its results.
* `test/commonmodeanalyzer_cfg.py` to monitor the common mode values in different detector parts. It requires the parameter `globalTag` in the command line.
* `test/TIDTECInnerRingInvestigator_cfg.py` to monitor the cluster occupancy and the digi (on cluster) distribution within the modules in the modules of the inner rings of TID and TEC
* `test/MultiplicityMonitor_cfg.py` to monitor the cluster multiplicity in the pixel and strip tracker detectors and their correlations

## ROOT Macros
This package contains a library of ROOT macros that can be loaded by executing `gSystem->Load("libDPGAnalysisSiStripToolsMacros.so")`
in the root interactive section. In case Root 6 is used the autocompletion of the macro names using the TAB is available only after a macro is executed.

### BSvsBPIXPlot
`BSvsBPIXPlot(TFile* ff, const char* bsmodule, const char* occumodule, const int run)`

It used the root file produced by `test/bsvsbpix_cfg.py` as input and produces a plot with the average beam spot position and the average BPIX ladder positions in the xy plane.

### ComputeOOTFractionvsFill
`ComputeOOTFractionvsFill(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm)`

It uses the output of `test/OOTmultiplicity_cfg.py` to produce a `OOTSummary` object which contains the trend of the out of time fraction vs the fill number.

### ComputeOOTFractionvsRun
`ComputeOOTFractionvsRun(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm=0)`

It uses the output of `test/OOTmultiplicity_cfg.py` to produce a `OOTSummary` object which contains the trend of the out of time fraction vs the run number.

### ComputerOOTFraction
`ComputeOOTFraction(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const int run, const char* hname, const bool& perFill=false)`
It uses the output of `test/OOTmultiplicity_cfg.py` to produce a `OOTResult` object with the result of the out of time fraction of a given run or fill.

### PlotOccupancyMap
`PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color)`

It requires as input a pointer to a root file, `ff`, which contains a `TDirectory` named `module`, produced by the 
`OccupancyPlots` EDAnalyzer (in this package). It produces histograms and color map of the cluster occupancy (number
of strips or pixels divided by the number of channels), multiplicity (number of clusters divided by the number of 
channels) in each subset of modules and the histograms of the ratio between the occupancy and the multiplicity which can
be interpreted as the cluster size. Each bin in the histograms represents a subset of modules represented in the
Tracker cross-section map. The parameters `min` and `max` are used to set the scale of the occupancy map while `mmin` and
`mmax` are ysed to se the scale of the multiplicity map. The parameter `color` is used to define the color palette
(color = 0 is the usual rainbow palette). If the macro is executed twice without deleting the previous TCanvas, the 
plots of the second execution are superimposed to the previous ones in the TCanvas with the six windows. This macro
depends on the setting of the file `DPGAnalysis/SiStripTools/python/occupancyplotsselection_simplified_cff`.

### PlotOccupancyMapPhase1
As the macro `PlotOccupancyMap` but for results obtained with the phase 1 geometry

### PlotOccupancyMapPhase2
As the macro `PlotOccupancyMap` but for results obtained with the phase 2 (Technical Proposal) geometry

### PlotOnTrackOccupancy
`PlotOnTrackOccupancy(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color)`

Similarly to the macro `PlotOccupancyMap` it requires as input a root file with the histograms produced by `OccupancyPlots`
but instead of one set of histograms it requires two of them in the TDirectory names `module` and `ontrkmod`. The macro
produces histograms and a Tracker cross-section map of the ratio of the average cluster multiplicity in the two set of
histograms. The purpose is to display the fraction of on-track clusters, for example.

### PlotOnTrackOccupancyPhase1
As the macro `PlotOnTrackOccupancy` but for results obtained with the phase 1 geometry

### PlotOnTrackOccupancyPhase2
As the macro `PlotOnTrackOccupancy` but for results obtained with the phase 2 geometry

### TrendPlotSingleBin
`TrendPlotSingleBin(TFile* ff, const char* module, const char* hname, const int bin)`

It has to be used with the output root file of `OccupancyPlots` and it produces a trend plot of the average occupancy or cluster multiplicity as a function of the run number for a specific part of the detector defined by the bin `bin` in the configuration of the `OccupancyPlots` job.

## Scripts
### merge_and_move_generic_fromCrab.sh
Script which helps to merge root files produced by a crab2 job and saved in the `res` sudirectory of output crab directory. 
### merge_and_move_generic_fromCrab3.sh
Script which helps to merge root files produced by a crab3 job and saved in the `results` sudirectory of output crab directory. 


