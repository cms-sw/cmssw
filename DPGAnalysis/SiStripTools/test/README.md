##Configurations

###OccupancyPlotsTest_cfg.py
this configurations produces cluster occupancy plots for different regions of the detector. A set of histograms is produced for each run.
The input should be a RECO file containing SiStripClusters, SiPixelClusters and a collection of tracks that is configurable. In case no RECO file is available for your purpose, it can be used also on RAW data setting the relevant option (fromRAW=1).
This configuration can be used on Cosmics or Collisions data.
To run on cosmics (RAW data) you can use the following command:

cmsRun OccupancyPlotsTest_cfg.py fromRAW=1 onCosmics=1 globalTag=<your-GT> inputFiles=<your-RAW-data-file>

To run on Collisions data you can use the following command:

cmsRun OccupancyPlotsTest_cfg.py fromRAW=1 onCosmics=0 trackCollection=generalTracks globalTag=<your-GT> inputFiles=<your-RAW-data-file>

there are other options that can be set in the command line:

1) "tag" add a suffix to the output file name
2) "maxEvents" to choose what is the number of events to be used (default -1)
3) "triggerPath" to choose the HLT path to filter your data (currently not working)
4) "HLTprocess" to choose which process to use for HLT selection default is "HLT"

NOTE: if you need to run on data older than 2016, the era is to be changed manually from "Run2_2016" to what reported in Configuration/StandardSequences/Eras.py

###crabOccupancy.py
is a sample crab3 configuration to run the above configuration on the grid. 