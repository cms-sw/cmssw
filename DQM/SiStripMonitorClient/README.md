###Macros and Scripts
####moduleOccupancyPlots
`moduleOccupancyPlots.sh <Datataking_period> <Dataset_type> <runnumber> <modulelistfile> <user certificate file> <user key file>`

This script produces a root file and png files of the occupancy plots of the modules selected with the file 
`<modulelistfile>` which contains a list of detid from run `<runnumber>` as found in the official DQM file. The 
`<Datataking_period>` and the `<Dataset_type>` have to match the names used in this site: 
https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/

To access the DQM file a valid certificate and key has to be provided to the script

####moduleOccupancyTrend
`moduleOccupancyTrend.sh <Datataking_period> <Dataset_type> <modulelistfile> <user certificate file> <user key file>` <runlistFile>

This script uses almost the same parameters as the moduleOccupancyPlot, instead of giving a single runNumber the run numbers are listed in the <runlistFile> a text file with a runNumber for each line. The script produces plots of single module occupancy superimposed for each run in the list, plus a trend of the occupancy per event, for single modules, and for the sum of all modules occupancy.