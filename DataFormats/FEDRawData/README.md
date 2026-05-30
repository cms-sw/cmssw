#  DataFormats/FEDRawData

## `FEDRawDataCollection`

The class `FEDRawDataCollection` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestFEDRawDataCollectionFormat` unit test, which makes use of the `TestReadFEDRawDataCollection` analyzer and the `TestWriteFEDRawDataCollection` producer. The unit test checks that the object can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of class `FEDRawDataCollection` gets changed in the future, please adjust the `TestReadFEDRawDataCollection` and `TestWriteFEDRawDataCollection` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-FEDRawData/](https://github.com/cms-data/DataFormats-FEDRawData/) repository, and update the `TestFEDRawDataCollectionFormat` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.

NOTE: FEDRawDataCollection is a legacy Run 3 format and will not be used for Phase-2.


## `RawDataBuffer`

The class `RawDataBuffer` is a new RAW data container intended for Phase-2 detector development. Key difference is that it enables 32-bit source IDs as only up to 4096 FED IDs are supported by `FEDRawDataCollection`.
Internally it uses a contiguous buffer, which is allocated by the input source once it knows full size of raw event data and prior to filling the collection with FED fragments. std::map is currently used to index offset and size of each sourceID block. Each ID block also include SlinkExpress header and trailer.

Classes RawDataBuffer and RawFragmentWrapper provide a thin API layer that will be stabilized and allow future changes to the internal data format. We aim to keep it backwards compatible as much as possible, but can not fully exclude breaking changes at the current point in development (as of LS3 start). This applies especially to API functions returning references to internal objects, such as std::map.

### `RawDataBuffer` API:

####  `const RawFragmentWrapper fragmentData(uint32_t sourceId)`
- **Description**: returns RawFragmentWrapper for a specific source ID. This class allows header, trailer and payload access

####  `const RawFragmentWrapper fragmentData(std::map<uint32_t, std::pair<uint32_t, uint32_t>>::const_iterator const& it)`
- **Description**: used with iterator from the internal std::map pointing to specific sourceID entry (for utilities manipulating or converting the whole collection). We reserve the possibility of changing this function in case of change of the internal format.

####  `unsigned char getByte(unsigned int pos)`
- **Description**: returns byte value at a specific position in the data buffer

####  std::vector<unsigned char> data() const { return data_; }
- **Description**: returns internal data buffer object. We reserve the possibility of changing this function.

####  std::map<uint32_t, std::pair<uint32_t, uint32_t>> const& map() const { return map_; }
- **Description**: returns internal source ID index map. We reserve the possibility of changing this function.

### `RawFragmentWrapper` API:

####  `std::span<const unsigned char> const& data()`
- **Description**: returns data byte span of the fragment wrapper (cotaining SLinkExpress header, payload and SLinkExpress trailer)

####  `std::span<const unsigned char> dataHeader(uint32_t expSize)`
- **Description**: retuens SLinkExpress header byte span. expectedSize is provided to check for consistency (at present this should be 16 bytes)

####  `std::span<const unsigned char> dataTrailer(uint32_t expSize)`
- **Description**: retuens SLinkExpress trailer byte span. expectedSize is provided to check for consistency (at present this should be 16 bytes)

####  `std::span<const unsigned char> payload(uint32_t expSizeHeader, uint32_t expSizeTrailer)`
- **Description**: returns payload byte span. expectedSize for header and trailer is provided to check for consistency (at present this should be 16 bytes each)

####  `uint32_t size()`
- **Description**: returns fragment size with header and trailer

####  `uint32_t sourceId()`
- **Description**: returns source ID of the wrapper object

####  `bool isValid()`
- **Description**: returns validity flag indicating that the wrapper was constructed on the actual data buffer
