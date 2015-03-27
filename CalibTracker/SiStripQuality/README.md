## Useful Tools
### Configurations
The following configuration files are examples of how to use the EDAnalyzer `SiStripQualityStastistics`. They can be used as they are, profiting of the fact that parameters can be passed in the command line, or edited for more complicated cases.
* `CalibTracker/SiStripQuality/test/cfg/SiStripQualityStatistics_cfg.py` : it produces the SiStripQuality statistics of a SiStripQuality object obtained by combining several records from a GlobalTag. The command line parameters are: `globalTag=<globaltag name>` and `runNumber=<run number>`.
* `CalibTracker/SiStripQuality/test/cfg/StudyExample/SiStripQualityStatistics_singleTag_cfg.py` : it produces the SiStripQuality statistics of a single bad component DB tag. The command line parameters are" `tagName=<tag name>` and `runNumber=<run number>`.
* `CalibTracker/SiStripQuality/test/cfg/StudyExample/SiStripQualityStatistics_Cabling_cfg.py` : it produces the SiStripQuality statistics of the channels which are **not** present in the cabling in a given tag. The command line parameters are: `cablingTagName=<FED cabling tag name>` and `runNumber=<run number>`.
