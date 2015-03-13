###Macros and Scripts
####moduleOccupancyPlots
`moduleOccupancyPlots.sh <Datataking_period> <Dataset_type> <runnumber> <modulelistfile> <user certificate file> <user key file>`

This script produces a root file and png files of the occupancy plots of the modules selected with the file 
`<modulelistfile>` which contains a list of detid from run `<runnumber>` as found in the official DQM file. The 
`<Datataking_period>` and the `<Dataset_type>` have to match the names used in this site: 
https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/

To access the DQM file a valid certificate and key has to be provided to the script
