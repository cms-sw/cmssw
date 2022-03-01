Some data files are found in the external area, see `CMSSW_SEARCH_PATH` for its location.
The files are visible in the search path (`CMSSW_SEARCH_PATH`) assuming all the files are accessed with `edm::FileInPath`, instead of direct filesystem calls.
These files can be found under the same path, as they would be under the usual software area.

Examples:
   * `CondTools/SiStrip/data` is the location of the input file for the `SiStripApvGainFromASCIIFile_cfg.py` unit test
