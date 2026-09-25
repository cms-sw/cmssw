# convertToRaw


Convert raw data stored into one or more EDM files format (.root files) or Streamer format (.dat files) into the DAQ
format (.raw files) used as input by the HLT.

The default behaviour is to process a single luminosity section at a time, in order to support luminosity sections split
across multiple files and a limit on the number of events in each lumisection.
If neither of these features is needed (i.e. if lumisections are not split, and all events should be converted) the `-1`
or `--one-file-per-lumi` can be used to process all data with a single job, speeding up the conversion considerably.

```
usage: convertToRaw [-h] [-s TAG] [-o PATH] [-f EVENTS] [-l EVENTS] [-r [RUN:LUMI-RUN:LUMI]] [-v] [-1] FILES [FILES ...]

positional arguments:
  FILES                 input files in .root or .dat format

options:
  -h, --help            show this help message and exit
  -s TAG, --source TAG  name of the FEDRawDataCollection to be repacked into raw format (default: rawDataCollector)
  -o PATH, --output PATH
                        base path to store the output files; subdirectories based on the run number are automatically created (default: /home/fwyzard/src/cmssw/HLTrigger/Tools)
  -f EVENTS, --events_per_file EVENTS
                        split the output into files with at most EVENTS events (default: 100)
  -l EVENTS, --events_per_lumi EVENTS
                        process at most EVENTS events in each lumisection (default: 11655)
  -r [RUN:LUMI-RUN:LUMI], --range [RUN:LUMI-RUN:LUMI]
                        process only the runs and lumisections in the given range (default: all)
  -v, --verbose         print additional information while processing the input files (default: False)
  -1, --one-file-per-lumi
                        assume that lumisections are not split across files (and disable --events_per_lumi) (default: False)
```
