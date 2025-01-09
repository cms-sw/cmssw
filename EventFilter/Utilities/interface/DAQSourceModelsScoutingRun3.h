#ifndef EventFilter_Utilities_DAQSourceModelsScoutingRun3_h
#define EventFilter_Utilities_DAQSourceModelsScoutingRun3_h

#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include <sys/types.h>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <memory>
#include <vector>

class DataModeScoutingRun3 : public DataMode {
public:
  DataModeScoutingRun3(DAQSource* daqSource) : DataMode(daqSource) {}
  ~DataModeScoutingRun3() override {}
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  void fillSDSRawDataCollection(SDSRawDataCollection& rawData, char* buff, size_t len);

  //reuse FRD file and event headers
  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }
  uint32_t headerSize() const override { return edm::streamer::FRDHeaderVersionSize[detectedFRDversion_]; }
  bool versionCheck() const override { return detectedFRDversion_ <= edm::streamer::FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override {
    // get event size from the first data source (main)
    return events_[0]->size();
  }

  void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) override {
    std::vector<uint64_t> const& fileSizes = rawFile->fileSizes_;
    fileHeaderSize_ = rawFile->rawHeaderSize_;
    numFiles_ = fileSizes.size();

    // initalize vectors keeping tracks of valid orbits and completed blocks
    sourceValidOrbitPair_.clear();
    completedBlocks_.clear();
    for (unsigned int i = 0; i < fileSizes.size(); i++) {
      completedBlocks_.push_back(false);
    }

    //add offset address for each file payload
    dataBlockAddrs_.clear();
    dataBlockAddrs_.push_back(addr);
    dataBlockMaxAddrs_.clear();
    dataBlockMaxAddrs_.push_back(addr + fileSizes[0] - fileHeaderSize_);
    auto fileAddr = addr;
    for (unsigned int i = 1; i < fileSizes.size(); i++) {
      fileAddr += fileSizes[i - 1];
      dataBlockAddrs_.push_back(fileAddr);
      dataBlockMaxAddrs_.push_back(fileAddr + fileSizes[i] - fileHeaderSize_);
    }

    dataBlockMax_ = rawFile->currentChunkSize();
    blockCompleted_ = false;
    //set event cached as we set initial address here
    bool result = makeEvents();
    assert(result);
    eventCached_ = true;
    setDataBlockInitialized(true);
  }

  bool nextEventView(RawInputFile*) override;
  bool blockChecksumValid() override { return true; }
  bool checksumValid() override;
  std::string getChecksumError() const override;

  uint32_t run() const override {
    assert(!events_.empty());
    return events_[0]->run();
  }

  bool dataBlockCompleted() const override { return blockCompleted_; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return true; }
  void unpackFile(RawInputFile* file) {}

  bool dataBlockInitialized() const override { return dataBlockInitialized_; }

  void setDataBlockInitialized(bool val) override { dataBlockInitialized_ = val; };

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override { return; }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                            std::vector<int> const& numSources,
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
  unsigned char* dataBlockAddr_ = nullptr;
  std::vector<unsigned char*> dataBlockAddrs_;
  std::vector<unsigned char*> dataBlockMaxAddrs_;
  size_t dataBlockMax_ = 0;
  short numFiles_ = 0;
  bool dataBlockInitialized_ = false;
  bool blockCompleted_ = true;
  bool eventCached_ = false;
  std::vector<std::filesystem::path> buPaths_;
  std::vector<int> buNumSources_;

  // keep track of valid (=aligned) orbits from different data sources
  std::vector<std::pair<int, int>> sourceValidOrbitPair_;
  unsigned int currOrbit_ = 0xFFFFFFFF;

  std::vector<bool> completedBlocks_;
};

#endif  // EventFilter_Utilities_DAQSourceModelsScoutingRun3_h
