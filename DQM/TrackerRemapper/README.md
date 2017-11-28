# Tracker Remapper Tool

The tool to either remap existing DQM histograms onto Strip Detector layout or to analyze event files and put eveents into the right detector parts.

## Running basics

Should you need to run this tool use: `cmsRun ConfFile_cfg.py [option=value]`

Options can be either adjusted in the configuration file (`ConfFile_cfg.py`) or passed as a list of `key=value` pairs. In the latter case unspecified options are taken directly from `ConfFile_cfg.py`.

## Options

  1. `opMode` - mode of operation, allowed values:
    1. `0` or `MODE_ANALYZE` (in configuration) 
    2. `1` or `MODE_REMAP` (in configuration)
  2. `analyzeMode` - option used to choose what do you want to put inside the map if `MODE_ANALYZE` was chosen:
    1. `1` or `RECHITS` (in configuration) for TRACKS
    2. `2` or `DIGIS` (in configuration) for DIGIS
    3. `3` or `CLUSTERS` (in configuration) for CLUSTERS
  3. `eventLimit` - only relevant for `MODE_ANALYZE` controls how many events from the input should be processed; default value is `100`, put `-1` to process all events from the input
  4. `inputRootFile` - relative path to the file to process, it should be different type of file depending on the `opMode` set:
    1. `MODE_ANALYZE` - DQM root file containing regular SiStrip plots
    2. `MODE_REMAP` - root file with events to process
  4. `stripHistogram` - histogram name to look for when `opMode=MODE_REMAP` is set, default is `TkHMap_NumberValidHits`
  5. `src` - automaticly set based on your settings, change at your own risk
  6. `globalTag` - global tag (GT) to use, default is `92X_upgrade2017_realistic_v11`
  
## The output

Your output will be saved by default as `outputStrip.root` in your current working directory. This can be tuned in the `ConfFile_cfg.py` file.


