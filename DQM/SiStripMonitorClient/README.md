###Macros and Scripts
####moduleOccupancyPlots
`moduleOccupancyPlots.sh <Datataking_period> <Dataset_type> <runnumber> <modulelistfile> <user certificate file> <user key file>`

This script produces a root file and png files of the occupancy plots of the modules selected with the file 
`<modulelistfile>` which contains a list of detid from run `<runnumber>` as found in the official DQM file. The 
`<Datataking_period>` and the `<Dataset_type>` have to match the names used in this site: 
https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/

To access the DQM file a valid certificate and key has to be provided to the script

#### TkMap_ script_ automatic_DB
`TkMap_script_automatic_DB.sh <Dataset_type> <runNumber>`

The script produces a set of png trackerMaps, a list of the bad components found by the prompt calibration loop as well as lists of the modules which are BAD from quality tests and modules with the largest digi, cluster, off trackcluster occupancy for the `<Dataset_type>`specified by `<runNumber>`. The `<Dataset_type>`and `<runNumber>`must match the names used in this site:	https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/

##SiStripOfflineDQM 

Configure the TrackerMaps to be created by the SiStrip Offline DQM Client using `"TkMapOptions"`. VPSet. The clientloops over the entries of ` "TkMapOptions" ` and generates the trackerMap specified by the string `mapName`. The  mapName menu is:  
 
        -QTestAlarm 
        -FractionOfBadChannels
        -NumberOfCluster
        -NumberOfDigi
        -NumberOfOfffTrackCluster
        -NumberOfOnTrackCluster
        -StoNCorrOnTrack
        -NApvShots
        -MedianChargeApvShots

In case `mapName=QTestAlarm`, the client fills the tracker Map with QTest Alarms and SiStripQuality bad modules, it also produces a text file with the number of bad modules and a list of bad modules per partition. In all other cases the tracker Map is filled from Histograms.

Create a sorted list of the N modules with the largest digi, cluster, off track cluster occupancy when the corresponding tracker maps are produced by setting `TopModules` as True in the corresponding entry in `"TkMapOptions"`. To choose the number of modules to be sort use int parameter `numberTopModules`.
