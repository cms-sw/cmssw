
# DTH orbit/event unpacker for DAQSource

<br>
This patch implements unpacking of the the DTH data format by `DAQSource` into `FedRawDataCollection`.
It both generates and consumes files with DTH format.

#Run the unit test
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

Note on the file format: apart of parsing single DTH orbit dump, input source plugin is capable also of building events from multiple DTH orbit blocks, but for the same orbit they must come sequentially in the file. Source scans the file and will find all blocks with orbit headers from the same orbit number, until a different orbit number is found or EOF, then it proceeds to build events from them by starting from last DTH event fragment trailer in each of the orbits found. This is then iterated for the next set of orbit blocks with the same orbit number in the file until file is processed. This is also valid at the level of individual files for the striped mode.

Striped mode is now supported for DTH, see `startFU_ds_multi.py` script. Multiple input directories can be specified, and number of sources for each can be provided in NumStreams (sources) vector. 1 is assumed if not specified. if num streams is specified, streamIDs (sourceIDs) numbers need to be provided in corresponding vector. Finally, sourceIdenfier (if specified) specifies prefix before the source number.
Example file name for this mode is `run123456_ls000_index000000_source01234.raw` with corresponding zfilled spaces zeroed.

It is possible that another DAQ-specific header will be added to both file and per-orbit to better encapsulate data (similar is done for Run2/3 FRD files), to provide additional metadata to improve integrity and completeness checks after aggregation of data in DAQ. At present, only RAW DTH is supported by the "DTH" module.

# DAQ file formats
Documentation:
https://twiki.cern.ch/twiki/bin/view/CMS/FFFMetafileFormats


# DAQSOurce DataMode Class Interface Reference

## Overview

The `DataMode` class is an **abstract base class** defining the interface for defining buffering strategy and data unpacking for the DAQ DAQSource modular input source. Subclasses must implement the pure virtual methods to provide specific functionality.

---

## Constructor & Destructor

### `DataMode(DAQSource* daqSource)`
- **Description**: Constructs a `DataMode` object with a reference to a `DAQSource`. Should be passed from the inheriting class constructor.
- **Parameters**:
  - `daqSource`: Pointer to a `DAQSource` used for data acquisition.

### `virtual ~DataMode() = default`
- **Description**: Virtual destructor (default implementation).

---

## Public Member Functions

### Data and Event Handling
- **`virtual std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() = 0`**
  Creates and returns a collection of DAQ provenance helper objects created for DataFormat objects passed to the CMSSW event processing. 
  **Returns**: Reference to a vector of `shared_ptr` to `const edm::DaqProvenanceHelper`.

- **`virtual void readEvent(edm::EventPrincipal& eventPrincipal) = 0`**
  Unpacks data prepared in previous interfaces (for example, setting pointers to corresponding data blocks) into the provided `EventPrincipal` object.
  It can also unpack data such as TDCS record to provide or substitute event metadata. 
  **Parameters**:
  - `eventPrincipal`: Reference to the event container where data is stored.

- **`virtual void unpackFile(RawInputFile* file) = 0`**
  Callback used to prepare data structures for "readEvent" For models which can do early unpacking outside of the main CMSSW loop.
  **Parameters**:
  - `file`: Pointer to the `RawInputFile` to unpack.

- **`virtual bool nextEventView(RawInputFile*) = 0`**
  Advances to the next event view in the input data block.
  Note that some models trivially contain one event within the block.
  **Returns**: `true` if successful, `false` otherwise.

---

### Version and Checksum Management
- **`virtual int dataVersion() const = 0`**
  **Returns**: Returns stored detected event data format version

- **`virtual void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) = 0`**
  Performs detection of the event data version from the first event or orbit in the file.
  **Parameters**:
  - `fileBuf`: Pointer to the buffer containing file data.
  - `fileHeaderOffset`: Offset of the header found at the beginning of the file in some models

- **`virtual bool versionCheck() const = 0`**
  **Returns**: `true` if the current version is valid, `false` otherwise.

- **`virtual bool blockChecksumValid() = 0`**
  **Returns**: `true` if the current data block's checksum is valid if checkum checking is enabled in the source.

- **`virtual bool checksumValid() = 0`**
  **Returns**: `true` if the overall checksum is valid if checksum checking is enabled in the source.

- **`virtual std::string getChecksumError() const = 0`**
  **Returns**: A descriptive error message if checksum validation fails.

---

### Data Block Operations
- **`virtual uint32_t headerSize() const = 0`**
  **Returns**: Size of the event or orbit header in bytes.

- **`virtual uint64_t dataBlockSize() const = 0`**
  **Returns**: Size of the current data block which represents complete orbit or event depending on the model

- **`virtual void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) = 0`**
  Creates a view of the data block. Internally all data pointers are set to be able to extract events using nextEventView until completion of the block
  Note that some models trivially contain one event within the block.
  **Parameters**:
  - `addr`: Pointer to the memory location.
  - `rawFile`: Associated raw input file.

- **`virtual bool dataBlockCompleted() const = 0`**
  **Returns**: `true` if the current data block processing is complete.

- **`virtual bool dataBlockInitialized() const = 0`** 
  **Returns**: `true` if the data block is initialized.
  **See Also**: `setDataBlockInitialized(bool)`.

- **`virtual void setDataBlockInitialized(bool) = 0`**
  Sets the initialization state of the data block. 

---

### File Managementi
- **`virtual std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName, bool fileListMode) const = 0`**
  Defines supplementary files required for processing.
  **Parameters**:
  - `primaryName`: Name of the primary file.
  - `fileListMode`: Whether the system is in file list mode.
  **Returns**: A pair containing:
  - `bool`: Success status.
  - `vector<string>`: List of additional files.

- **`virtual bool isMultiDir() const`**
  **Returns**: `false` by default. Override to return `true` if models supports "striped" reading from multiple directories (sources)

- **`virtual void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::vector<int> const& numSources, std::vector<int> const& sourceIDs, std::string const& sourceIdentifier, std::string const& runDir) = 0`**
  Provided for multi-dir models to compose list of input directories and mapping to input files. Specification of number of individual sources (streams) is provided and, if defined, full list of source(stream) ID names. Source identifier will be non-empty if source suffix is used in the filename. runDir is run-specific input directory name. For single-directory models sufficient information is already provided with other APIs.
  **Parameters**:
  - `baseDirs`: List of base directories for output.
  - `numSources`: Number of data sources per input directory.
  - `sourceIDs`: List of source(stream) IDs.
  - `sourceIdentifier`: Identifier for the source suffix.
  - `runDir`: Name of the input run directory.

---

### Configuration and State
- **`virtual void setTCDSSearchRange(uint16_t, uint16_t) = 0`**
  Sets the search range for TCDS FED/SourceID (Trigger and Clock Distribution System) in models where relevant.

- **`bool errorDetected()`**
  **Returns**: `true` if an error was detected.

- **`void setTesting(bool testing)`**
  Enables/disables testing mode.
  **Parameters**:
  - `testing`: `true` to enable testing mode.

---

### Run and Event Information
- **`virtual uint32_t run() const = 0`**
  **Returns**: The current run number detected by the model.

- **`virtual bool hasEventCounterCallback() const`**
  **Returns**: `false` by default. Override to enable file event counter for using raw files without file header in the live mode.

- **`virtual int eventCounterCallback(std::string const& name, int& fd, int64_t& fsize, uint32_t sLS, bool& found) const`**
  Callback for pre-parsing files to count events in case files do not have file header providing this information.
  **Returns**: `-1` by default (no action). Override to implement counting.

