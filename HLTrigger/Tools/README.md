# convertToRaw

Convert RAW data stored in one or more EDM .root files into the .raw file used as input by the HLT.

```
usage: convertToRaw [-h] [-o PATH] [-f EVENTS] [-l EVENTS] [--one-file-per-lumi] FILES [FILES ...]

Convert RAW data from .root format to .raw format.

positional arguments:
  FILES                 input files in .root format

optional arguments:
  -h, --help            show this help message and exit
  -o PATH, --output PATH
                        base path to store the output files; subdirectories based on the run number are automatically created (default: )
  -f EVENTS, --events_per_file EVENTS
                        split the output into files with at most EVENTS events (default: 50)
  -l EVENTS, --events_per_lumi EVENTS
                        process at most EVENTS events in each lumisection (default: 11650)
  --one-file-per-lumi   assume that lumisections are not split across files (and disable --events_per_lumi) (default: False)
```

The default behaviour is to process a single luminosity section at a time, in order to support luminosity sections split across multiple files and a limit on the number of events in each lumisection.

If neither of these features is needed (_i.e._ if lumisections are not split, and all events should be converted) the `--one-file-per-lumi` can be used to process all data with a single job, speeding up the conversion considerably.
