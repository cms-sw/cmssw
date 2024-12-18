
# DTH orbit/event unpacker for DAQSource

https://github.com/smorovic/cmssw/tree/15_0_0_pre1-source-improvements
<br>
This patch implements unpacking of the the DTH data format by `DAQSource` into `FedRawDataCollection`.

It is rebased over CMSSW master (compatible with 15_0_0_pre1 at the time this file is commited), but it builds and runs in 14_2_0 as well. All changes are contained in `EventFilter/Utilities`.

## Fetching the code

```
scram project CMSSW_15_0_0_pre1 #or CMSSW_14_2_0 (currently it compiles and runs also in 14_X releases)
git cms-addpkg EventFilter/Utilities
git remote add smorovic https://github.com/smorovic/cmssw.git
git fetch smorovic 15_0_0_pre1-source-improvements:15_0_0_pre1-source-improvements
git checkout 15_0_0_pre1-source-improvements
scram b
```

Run the unit test (generates and consumes files with DTH format):
```
cmsenv
cd src/EventFilter/Utilities/test
./RunBUFU.sh
```

## Important code and scripts in `EventFilter/Utilities`:

Definition of DTH orbit header, fragment trailer and SLinkRocket header/trailer (could potentially be moved to DataFormats or another package in the future):
<br>
[interface/DTHHeaders.h](../interface/DTHHeaders.h)

Plugin for DAQSource (input source) which parses the DTH format:
<br>
[src/DAQSourceModelsDTH.cc](../src/DAQSourceModelsDTH.cc)

Generator of dummy DTH payload for the fake "BU" process used in unit tests:
<br>
[plugins/DTHFakeReader.cc](../plugins/DTHFakeReader.cc)

Script which runs the unit test with "fakeBU" process generating payload from multiple DTH sources (per orbit) and "FU" CMSSW job consuming it:
<br>
[test/testDTH.sh](../test/testDTH.sh)

FU cmsRun configuration used in above tests:
<br>
[test/unittest_FU_daqsource.py](../test/unittest_FU_daqsource.py)

## Running on custom input files
`unittest_FU_daqsource.py` script can be used as a starting point to create a custom runner with inputs such as DTH dumps (not generated as in the unit test). DAQSource should be set to `dataMode = cms.untracked.string("DTH")` to process DTH format. Change `fileListMode` to `True` and fill in `fileList` parameter with file paths to run with custom files, however they should be named similarly and could also be placed in similar directory structure, `ramdisk/runXX`, to provide initial run and lumisection to the source. Run number is also passed to the source via the command line as well as the working directory (see `testDTH.sh` script).

Note on the file format: apart of parsing single DTH orbit dump, input source plugin is capable also of building events from multiple DTH orbit blocks, but for the same orbit they must come sequentially in the file. Source scans the file and will find all blocks with orbit headers from the same orbit number, until a different orbit number is found or EOF, then it proceeds to build events from them by starting from last DTH event fragment trailer in each of the orbits found. This is then iterated for the next set of orbit blocks with the same orbit number in the file until file is processed.

It is possible that another DAQ-specific header will be added to both file and per-orbit to better encapsulate data similar is done for Run2/3 RAW data), to provide additional metadata to improve integrity and completeness checks after aggregation of data in DAQ. At present, only RAW DTH is supported by "DTH" format.
