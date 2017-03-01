## Scripts
### `MonitorDB_NewDirStructure_KeepTagLinks_generic_V2.sh`
This scripts is used by the cronjob to monitor the SiStrip DB tags described here:
https://twiki.cern.ch/twiki/bin/view/CMS/StripTrackerMonitoringCondDb

It discovers all SiStrip related tags in a given DB account, for example pro/CMS_CONDITIONS, with a command like `conddb --db pro  listTags` Be careful: tags with the string "V0" in their names are ignored to avoid to analyze some tags, duplicate of other tags.
It creates a directory for the DB under investigation (e.g. pro) and a subdirectory for the account (e.g. CMS_CONDITIONS). In this subdirectory further subdirectories are created for the strip tags (e.g. DBTagCollection/SiStripTagA, DBTagCollection/SiStripTagB, DBTagCollection/SiStripTagC ...).
It retrieves the list of IOVs for the strip tag under investigation using a command like `conddb --db pro list -L 5000 $tag`
For each IOV, the database values (for example the noise values) are retrieved from the DB, filled in summary histograms and stored both in root files (in the subdirectory rootfiles) and in png format (in the subdirectory plots). This is done using `test/DBReader_conddbmonitoring_generic_cfg.py`.
Finally, the png pictures are published on the web.

### `MonitorDB_NewDirStructure_KeepTagLinks_generic.sh`
As the script above but compatible with conddb V1

### `Monitor_GlobalTags_V2.sh`
This scripts is used by the cronjob to monitor the SiStrip DB tags described here:
https://twiki.cern.ch/twiki/bin/view/CMS/StripTrackerMonitoringCondDb

It creates a directory named GlobalTags and a set of subdirectories for each Global Tag.
It loops on all the known Global Tags in the chosen database (e.g. pro) and to look for all the SiStrip database tags in each one of them to create a set of bi-directional links in files in the subdirectories RelatedGlobalTags in the tag directories and in the Global Tag subdirectories.

### `Monitor_GlobalTags.sh`
As the script above but compatible with conddb V1

### `Monitor_NoiseRatios_V2.sh`
This scripts is used by the cronjob to monitor the SiStrip DB tags described here:
https://twiki.cern.ch/twiki/bin/view/CMS/StripTrackerMonitoringCondDb

For each GlobalTag the pair of SiStripNoiseRcd and SiStripApvGainRcd is looked for and the ratio of the normalized strip noise of each IOV w.r.t. to the previous one is computed using the plugin `SiStripNoiseCorrelate` with the configuration file `test/SiStripCorrelateNoise_conddbmonitoring_cfg.py`. The average of the ratio in each module is plotted in a tracker map as well as the distributions of the values in each subdetector.

### `Monitor_NoiseRatios.sh`
As the script above but compatible with conddb V1

### `makeModulePlots.sh`
It executes the CMSSW job to produce the noise and/or pedestal histograms for a selected list of modules (detid) and then produce the png files with the plots. It requires as input: the run number, the name of the file with the list of detid, a two-bit number to define whether pedestal or noise values are requested, the name of the GlobalTag, the DB connection string, the tag name and the record name (if the GlobalTag is not provided), a boolean which is true if the gain normalization has to be applied and a boolean which is true if the gain to be used in the one used in the simulation (it applies only to MC GlobalTags)

## Configuration files
### `test/DBReader_conddbmonitoring_generic_cfg.py`
This configuration uses the plugins `SiStripMonitorCondData`, `SistripQualityStatistics`, `SiStripLatencyDummyPrinter` and `SiStripConfObjectDummyPrinter` to monitor the SiStrip database conditions. An example of the command to be executed is the following:

`cmsRun DQM/SiStripMonitorSummary/test/DBReader_conddbmonitoring_generic_cfg.py print logDestination=cout qualityLogDestination=QualityInfo cablingLogDestination=CablingInfo condLogDestination=Dummy outputRootFile=SiStripNoise_GR10_v1_hlt_Run_211725.root connectionString=frontier://PromptProd/CMS_COND_31X_STRIP recordName=SiStripNoisesRcd recordForQualityName=Dummy tagName=SiStripNoise_GR10_v1_hlt runNumber=211725 LatencyMon=False ALCARecoTriggerBitsMon=False ShiftAndCrosstalkMon=False APVPhaseOffsetsMon=False PedestalMon=False NoiseMon=True QualityMon=False CablingMon=False GainMon=False LorentzAngleMon=False BackPlaneCorrectionMon=False ThresholdMon=False MonitorCumulative=True ActiveDetId=True`

where the meaning of each parameter is the following:
* `logDestination`: name of the file where the error and warning messages are written. It can be cout
* `qualityLogDestination`: name of the file where the summary of the list of bad modules is written. Used if the option QualityMon = True
* `cablingLogDestination`: name of the file where the dump of the cabling is written. Used if the option CablingMon = True
* `condLogDestination`: name of the file where conditions monitoring summaries are written. Used if the options LatencyMon = True or ShiftAndCrosstalkMon 0 True
* `outputRootFile`: name of the output root file with the monitoring histograms
* `connectionString`: name of the string to connect to the proper database account
* `recordName`: name of the record which contains the database tag to be monitored
* `recordForQualityName`: it has to be equal to SiStripDetCablingRcd when the option CablingMon = True to prepare the list of disconnected channels from a given cabling
* `tagName`: name of the database tag to be monitored
* `runNumber`: run number used to define the IOV to be monitored
* `LatencyMon`: True if a Latency tag has to be monitored
* `ALCARecoTriggerBitsMon`: True if an ALCARECO trigger bit tag has to be monitored
* `ShiftAndCrosstalkMon`: True if a back plane correction and cross talk tag has to be monitored
* `APVPhaseOffsetsMon`: True if a APV cycle phase offset tag has to be monitored
* `PedestalMon`: True if a pedestal tag has to be monitored
* `NoiseMon`: True if a noise tag gas to be monitored
* `QualityMon`: True if a bad channel tag has to be monitored
* `CablingMon`: True if a cabling tag has to be monitored
* `GainMon`: True if an APV gain tag has to be monitored
* `LorentzAngleMon`: True if a Lorentz angle tag has to be monitored
* `BackPlaneCorrectionMon`: True if a (new type) back plane correction tag has to be monitored
* `ThresholdMon`: True if a threshold tag has to be monitored
* `MonitorCumulative`: set always to True
* `ActiveDetid`: set always to True
Remarks:
* The configuration used in the command above is designed to monitor a single tag at a time and the tag and record name are provided explicitly as inputs. In case tags from a GlobalTag have to be monitored the configuration has to be modified accordingly, for example by replacing this module: `process.poolDBESSource` with lines like:
`process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") 
process.GlobalTag.globaltag = < global tag name >`
* All the parameters above have a default value which can be found by looking at the content of the file. Therefore there is no need to provide all of them.

### `test/DBReader_conddbmonitoring_singlemodule_cfg.py`
This configuration file is used to produce DQM root files with the pedestal and/or noise histograms of selected modules (detid) and selected DB tags or GlobalTags and IOV. An example of the command is the following:
`cmsRun test/DBReader_conddbmonitoring_singlemodule_cfg.py logDestination=cout outputRootFile=$outputroot moduleList_load=<module list file name> globalTag=<globaltag name> connectionString=<db connection string> tagName=<DB tag name> recordName=<record name> runNumber=<run number> PedestalMon=True/False NoiseMon=True/False gainNorm=True/False simGainNorm=True/False`
where the `connectionString` the `tagName` and the `recordName` are needed if no GlobalTag is provided and the booleans are used to:
 * `PedestalMon`: if the pedestal values are requested
 * `NoiseMon`: if the noise values are requested
 * `gainNorm`: if the noise has to be normalized for the gain
 * `simGainNorm`: if the noise has to be normalized for the gain used in the simulation

### `test/SiStripCorrelateNoise_conddbmonitoring_cfg.py`
This configuration is used to compare, with the plugin `SiStripCorrelateNoise` which computes the ratio of the average module noise, the different IOVs of a SiStripNoise database tag. Before computing the ratio the noise values are normalized using the APV gain values which are provided with a SiStripApvGainRcd record. The following cmsRun command is an example of how to use it:
`cmsRun DQM/SiStripMonitorSummary/test/SiStripCorrelateNoise_conddbmonitoring_cfg.py print connectionString=frontier://PromptProd/CMS_COND_31X_STRIP noiseTagName=SiStripNoise_GR10_v1_hlt gainTagName=SiStripApvGain_GR10_v1_hlt firstRunNumber=211591 secondRunNumber=211725`
where the meaning of each parameter is the following:
* `connectionString`: name of the string to connect to the proper database account
* `noiseTagName`: name of the strip noise database tag to be monitored
* `gainTagName`: name of the strip APV gain database tag name to be used for the noise normalization
* `firstRunNumber`: run number from which the first IOV (denominator) is extracted
* `secondRunNumber`: run number from the the second IOV (numerator) is extracted

## Binaries
### `makeModulePlots`
Extract and print in png files the noise or pedestal values of a list of selected modules (detid) from a DQM-like root file produced by `SiStripMonitorCondData`. It requires as arguments: the input root file name, the name of the file with the list of detid, a two-bit integer to select whether pedestal or noise histograms have to be extracted, the name of the directory where the png files are saved and a suffix to be appended to the png file names.
### 
