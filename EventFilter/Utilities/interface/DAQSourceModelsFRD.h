#ifndef EventFilter_Utilities_DAQSourceModelsFRD_h
#define EventFilter_Utilities_DAQSourceModelsFRD_h

#include <filesystem>

#include "EventFilter/Utilities/interface/DAQSourceModels.h"

class FEDRawDataCollection;

class DataModeFRD : public DataMode {
public:
  DataModeFRD(DAQSource* daqSource) : DataMode(daqSource) {}
  ~DataModeFRD() override{};
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

  uint32_t headerSize() const override { return FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override { return event_->size(); }

  void makeDataBlockView(unsigned char* addr,
                         size_t maxSize,
                         std::vector<uint64_t> const& fileSizes,
                         size_t fileHeaderSize) override {
    dataBlockAddr_ = addr;
    dataBlockMax_ = maxSize;
    eventCached_ = false;
    nextEventView();
    eventCached_ = true;
  }

  bool nextEventView() override;
  bool checksumValid() override;
  std::string getChecksumError() const override;

  bool isRealData() const override { return event_->isRealData(); }

  uint32_t run() const override { return event_->run(); }

  //true for DAQ3 FRD
  bool dataBlockCompleted() const override { return true; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return false; }
  bool dataBlockInitialized() const override { return true; }

  void setDataBlockInitialized(bool) override{};

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {
    MINTCDSuTCAFEDID_ = MINTCDSuTCAFEDID;
    MAXTCDSuTCAFEDID_ = MAXTCDSuTCAFEDID;
  }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) override {}

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName, bool) const override {
    return std::make_pair(true, std::vector<std::string>());
  }

private:
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  uint16_t detectedFRDversion_ = 0;
  size_t headerSize_ = 0;
  std::unique_ptr<FRDEventMsgView> event_;
  uint32_t crc_ = 0;
  unsigned char* dataBlockAddr_ = nullptr;
  size_t dataBlockMax_ = 0;
  size_t fileHeaderSize_ = 0;
  uint16_t MINTCDSuTCAFEDID_ = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID_ = FEDNumbering::MAXTCDSuTCAFEDID;
  bool eventCached_ = false;
};

/* 
 * Demo for FRD source reading files from multiple striped destinations
 *
 * */

class DataModeFRDStriped : public DataMode {
public:
  DataModeFRDStriped(DAQSource* daqSource) : DataMode(daqSource) {}
  ~DataModeFRDStriped() override{};
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  //non-virtual
  edm::Timestamp fillFRDCollection(FEDRawDataCollection& rawData, bool& tcdsInRange, unsigned char*& tcds_pointer);

  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }

  uint32_t headerSize() const override { return FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override {
    //just get first event size
    if (events_.empty())
      throw cms::Exception("DataModeFRDStriped::dataBlockSize") << " empty event array";
    return events_[0]->size();
  }

  void makeDataBlockView(unsigned char* addr,
                         size_t maxSize,
                         std::vector<uint64_t> const& fileSizes,
                         size_t fileHeaderSize) override {
    fileHeaderSize_ = fileHeaderSize;
    numFiles_ = fileSizes.size();
    //add offset address for each file payload
    dataBlockAddrs_.clear();
    dataBlockAddrs_.push_back(addr);
    dataBlockMaxAddrs_.clear();
    dataBlockMaxAddrs_.push_back(addr + fileSizes[0] - fileHeaderSize);
    auto fileAddr = addr;
    for (unsigned int i = 1; i < fileSizes.size(); i++) {
      fileAddr += fileSizes[i - 1];
      dataBlockAddrs_.push_back(fileAddr);
      dataBlockMaxAddrs_.push_back(fileAddr + fileSizes[i] - fileHeaderSize);
    }

    dataBlockMax_ = maxSize;
    blockCompleted_ = false;
    //set event cached as we set initial address here
    bool result = makeEvents();
    assert(result);
    eventCached_ = true;
    setDataBlockInitialized(true);
  }

  bool nextEventView() override;
  bool checksumValid() override;
  std::string getChecksumError() const override;

  bool isRealData() const override {
    assert(!events_.empty());
    return events_[0]->isRealData();
  }

  uint32_t run() const override {
    assert(!events_.empty());
    return events_[0]->run();
  }

  bool dataBlockCompleted() const override { return blockCompleted_; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return true; }

  bool dataBlockInitialized() const override { return dataBlockInitialized_; }

  void setDataBlockInitialized(bool val) override { dataBlockInitialized_ = val; };

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {
    MINTCDSuTCAFEDID_ = MINTCDSuTCAFEDID;
    MAXTCDSuTCAFEDID_ = MAXTCDSuTCAFEDID;
  }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) override;

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName,
                                                                  bool fileListMode) const override;

private:
  bool makeEvents();
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;  //
  uint16_t detectedFRDversion_ = 0;
  size_t fileHeaderSize_ = 0;
  size_t headerSize_ = 0;
  std::vector<std::unique_ptr<FRDEventMsgView>> events_;
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
};

#endif  // EventFilter_Utilities_DAQSourceModelsFRD_h
