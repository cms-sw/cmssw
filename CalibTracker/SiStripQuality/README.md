## Code description 
There are three algorithms available in this package which are used to find Bad APVs and strips for the SiStrip Tracker.
* `SiStripBadAPVAlgorithmFromClusterOccupancy` : as the name suggests it looks for the Bad APVs in individual layers/discs of the tracker. It first finds the mean and RMS of all the APV medians in a given layer/disc using a few iterations excluding 3 sigma outliers. Then these APV medians are individually required to be within N(configurable) times the calculated RMS. Otherwise the APV is marked bad. 
* `SiStripHotStripAlgorithmFromClusterOccupancy` : it identifies bad strips using Poisson probability using a few iterations till no new bad strips are detected. At the starting point of each iteration average entries per strip is calculated excluding bad strips and then Poissonian is estimated using this average. If the estimated Poissonian is greater than the predefined (configurable) probablity the strip is marked hot. 
* `SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy` : it's a combination of above two algorithms to identify Bad APVs and Hot Strips in one go. This is used in the official Prompt Calibration Loop.

## Useful Tools
### Configurations
The following configuration files are examples of how to use the EDAnalyzer `SiStripQualityStastistics`. They can be used as they are, profiting of the fact that parameters can be passed in the command line, or edited for more complicated cases.
* `CalibTracker/SiStripQuality/test/cfg/SiStripQualityStatistics_cfg.py` : it produces the SiStripQuality statistics of a SiStripQuality object obtained by combining several records from a GlobalTag. The command line parameters are: `globalTag=<globaltag name>` and `runNumber=<run number>`.
* `CalibTracker/SiStripQuality/test/cfg/StudyExample/SiStripQualityStatistics_singleTag_cfg.py` : it produces the SiStripQuality statistics of a single bad component DB tag. The command line parameters are" `tagName=<tag name>` and `runNumber=<run number>`.
* `CalibTracker/SiStripQuality/test/cfg/StudyExample/SiStripQualityStatistics_Cabling_cfg.py` : it produces the SiStripQuality statistics of the channels which are **not** present in the cabling in a given tag. The command line parameters are: `cablingTagName=<FED cabling tag name>` and `runNumber=<run number>`.
