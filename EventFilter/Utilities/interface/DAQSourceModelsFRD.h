#ifndef EventFilter_Utilities_DAQSourceModelsFRD_h
#define EventFilter_Utilities_DAQSourceModelsFRD_h

/*
* DAQSource data model classes for reading Run3 FRD format and unpacking into the FedRawDataCollection
* FRD: standard readout of input from the event builder
* FRDPreUNpack: variant unpacking events tns nto FedRawDataCollection class in reader threads
* FRSStiped: more generic version able to read from multiple source
* directories (Super-Fragmeng Builder DAQ)
* */

#include <filesystem>
#include <queue>
#include "oneapi/tbb/concurrent_unordered_set.h"

#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class FEDRawDataCollection;

/*
 * FRD unpacker equivalent to the FedRawDataInputSource
 */

class DataModeFRD : public DataMode {
public:
  DataModeFRD(DAQSource* daqSource, bool verifyFEDs) : DataMode(daqSource), verifyFEDs_(verifyFEDs) {}
  DataModeFRD(DAQSource* daqSource) : DataMode(daqSource) {}
  ~DataModeFRD() override {}
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  //non-virtual
  edm::Timestamp fillFEDRawDataCollection(FEDRawDataCollection& rawData,
                                          bool& tcdsInRange,
                                          unsigned char*& tcds_pointer);

  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }

  uint32_t headerSize() const override { return edm::streamer::FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= edm::streamer::FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override { return event_->size(); }

  void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) override;
  bool nextEventView(RawInputFile*) override;
  bool blockChecksumValid() override { return true; }
  bool checksumValid() override;
  std::string getChecksumError() const override;

  //bool isRealData() const override { return event_->isRealData(); }

  uint32_t run() const override { return event_->run(); }

  //true for DAQ3 FRD
  bool dataBlockCompleted() const override { return true; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return false; }

  void unpackFile(RawInputFile*) override {}

  bool dataBlockInitialized() const override { return true; }

  void setDataBlockInitialized(bool) override {}

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {
    MINTCDSuTCAFEDID_ = MINTCDSuTCAFEDID;
    MAXTCDSuTCAFEDID_ = MAXTCDSuTCAFEDID;
  }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                            std::vector<int> const& numSources,
                            std::vector<int> const& sourceIDs,
                            std::string const& sourceIdentifier,
                            std::string const& runDir) override {}

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName, bool) const override {
    return std::make_pair(true, std::vector<std::string>());
  }

private:
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  uint16_t detectedFRDversion_ = 0;
  size_t headerSize_ = 0;
  std::unique_ptr<edm::streamer::FRDEventMsgView> event_;
  uint32_t crc_ = 0;
  unsigned char* dataBlockAddr_ = nullptr;
  size_t dataBlockMax_ = 0;
  size_t fileHeaderSize_ = 0;
  uint16_t MINTCDSuTCAFEDID_ = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID_ = FEDNumbering::MAXTCDSuTCAFEDID;
  bool eventCached_ = false;
  std::unordered_set<unsigned short> fedIdSet_;
  unsigned int expectedFedsInEvent_ = 0;
  bool verifyFEDs_ = true;
};

/*
 * FRD source prebuffering in the reader thread
 */

class DataModeFRDPreUnpack : public DataMode {
public:
  DataModeFRDPreUnpack(DAQSource* daqSource, bool verifyFEDs) : DataMode(daqSource), verifyFEDs_(verifyFEDs) {}
  ~DataModeFRDPreUnpack() override {};
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  //non-virtual
  void unpackEvent(edm::streamer::FRDEventMsgView* eview, UnpackedRawEventWrapper* ec, unsigned int ls);
  void unpackFile(RawInputFile*) override;
  edm::Timestamp fillFEDRawDataCollection(edm::streamer::FRDEventMsgView* eview,
                                          FEDRawDataCollection& rawData,
                                          bool& tcdsInRange,
                                          unsigned char*& tcds_pointer,
                                          bool& err,
                                          std::string& errmsg);

  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }

  uint32_t headerSize() const override { return edm::streamer::FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= edm::streamer::FRDHeaderMaxVersion; }

  //used
  uint64_t dataBlockSize() const override { return event_->size(); }

  void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) override;
  bool nextEventView(RawInputFile*) override;
  bool blockChecksumValid() override { return true; }
  bool checksumValid() override;
  std::string getChecksumError() const override;

  uint32_t run() const override { return ec_->run(); }

  //true for DAQ3 FRD
  bool dataBlockCompleted() const override { return true; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return true; }

  bool dataBlockInitialized() const override { return true; }

  void setDataBlockInitialized(bool) override {};

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {
    MINTCDSuTCAFEDID_ = MINTCDSuTCAFEDID;
    MAXTCDSuTCAFEDID_ = MAXTCDSuTCAFEDID;
  }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                            std::vector<int> const& numSources,
                            std::vector<int> const& sourceIDs,
                            std::string const& sourceIdentifier,
                            std::string const& runDir) override {}

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName, bool) const override {
    return std::make_pair(true, std::vector<std::string>());
  }

private:
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  uint16_t detectedFRDversion_ = 0;
  size_t headerSize_ = 0;
  std::unique_ptr<edm::streamer::FRDEventMsgView> event_;
  std::unique_ptr<UnpackedRawEventWrapper> ec_;
  uint32_t crc_ = 0;
  unsigned char* dataBlockAddr_ = nullptr;
  size_t dataBlockMax_ = 0;
  size_t fileHeaderSize_ = 0;
  uint16_t MINTCDSuTCAFEDID_ = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID_ = FEDNumbering::MAXTCDSuTCAFEDID;
  bool eventCached_ = false;
  oneapi::tbb::concurrent_unordered_set<unsigned short> fedIdSet_;
  std::atomic<unsigned int> expectedFedsInEvent_ = 0;
  bool verifyFEDs_ = true;
};

/* 
 * FRD source reading files from multiple striped destinations (Super-Fragment Builder DAQ)
 *
 * */

class DataModeFRDStriped : public DataMode {
public:
  DataModeFRDStriped(DAQSource* daqSource, bool verifyFEDs) : DataMode(daqSource), verifyFEDs_(verifyFEDs) {}
  ~DataModeFRDStriped() override {}
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  //non-virtual
  edm::Timestamp fillFRDCollection(FEDRawDataCollection& rawData, bool& tcdsInRange, unsigned char*& tcds_pointer);

  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }

  uint32_t headerSize() const override { return edm::streamer::FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= edm::streamer::FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override {
    //just get first event size
    if (events_.empty())
      throw cms::Exception("DataModeFRDStriped::dataBlockSize") << " empty event array";
    return events_[0]->size();
  }

  void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) override;
  bool nextEventView(RawInputFile*) override;
  bool blockChecksumValid() override { return true; }
  bool checksumValid() override;
  std::string getChecksumError() const override;

  //bool isRealData() const override {
  //  assert(!events_.empty());
  //  return events_[0]->isRealData();
  //}

  uint32_t run() const override {
    assert(!events_.empty());
    return events_[0]->run();
  }

  bool dataBlockCompleted() const override { return blockCompleted_; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return true; }

  void unpackFile(RawInputFile*) override {}

  bool dataBlockInitialized() const override { return dataBlockInitialized_; }

  void setDataBlockInitialized(bool val) override { dataBlockInitialized_ = val; }

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {
    MINTCDSuTCAFEDID_ = MINTCDSuTCAFEDID;
    MAXTCDSuTCAFEDID_ = MAXTCDSuTCAFEDID;
  }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                            std::vector<int> const& numSources,
                            std::vector<int> const& sourceIDs,
                            std::string const& sourceIdentifier,
                            std::string const& runDir) override;

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName,
                                                                  bool fileListMode) const override;

private:
  bool makeEvents();
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;  //
  uint16_t detectedFRDversion_ = 0;
  size_t fileHeaderSize_ = 0;
  size_t headerSize_ = 0;
  std::vector<std::unique_ptr<edm::streamer::FRDEventMsgView>> events_;
  std::string crcMsg_;
  unsigned char* dataBlockAddr_ = nullptr;
  std::vector<unsigned char*> dataBlockAddrs_;
  std::vector<unsigned char*> dataBlockMaxAddrs_;
  size_t dataBlockMax_ = 0;
  short numFiles_ = 0;
  bool dataBlockInitialized_ = false;
  bool blockCompleted_ = true;
  bool eventCached_ = false;
  uint16_t MINTCDSuTCAFEDID_ = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID_ = FEDNumbering::MAXTCDSuTCAFEDID;
  std::vector<std::filesystem::path> buPaths_;
  std::unordered_set<unsigned short> fedIdSet_;
  unsigned int expectedFedsInEvent_ = 0;
  bool verifyFEDs_ = true;
};

#endif  // EventFilter_Utilities_DAQSourceModelsFRD_h
