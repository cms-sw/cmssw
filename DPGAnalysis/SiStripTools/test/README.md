##Configurations

###OccupancyPlotsTest_cfg.py
This configuration can be used to measure the cluster and digi on cluster occupancy in different parts of the Tracker detectors: the detector is divided in several subsets of modules which are more or less in the same r and z position. This configuration makes use of the `EDAnalyzer`s `OccupancyPlots`, `MultiplicityInvestigator`, `TrackCount` and `EventTimeDistribution`.
The command to run it is:
`cmsRun OccupancyPlotsTest_cfg.py withTracks=0/1 globalTag=<your-GT> inputFiles=<your-RAW-data-file> tag=<suffix-to-the-root-file-name>`
If the option `withTracks=1` then the cluster and digi occupancy is measured also for the on-track clusters. If the input files are RAW data the options `fromRAW=1` has to be used (and `withTracks=0`). To select a specific HLT path the option `triggerPath` can be used (for example `triggerPath="HLT_ZeroBias_v*"`. In this case the histograms will be produced for all the processed events and only for the events which fulfill the trigger selection. With the option `maxEvents=<number-of-events>` it is possible to run on a limited number of events.


###crabOccupancy.py
is a sample crab3 configuration to run the above configuration on the grid. 
