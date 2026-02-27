#ifndef EventFilter_Utilities_DAQSourceModelsDTH_h
#define EventFilter_Utilities_DAQSourceModelsDTH_h

/*
 * DAQ Source module for DTH readout
 * Used by modular DAQSource to read files containing raw DTH orbit payload.
 * Orbits are unpacked into individual events which are queued to the framework as FedRawDataCollection object.
 * If more than one sourceID blocks is included they will all be unpacked, they need to be adjacent in the file
 * for the same orbit. Exception to this is reading from multiple file sources,
 * in that case that is only required locally in a file, but orbits need to come in the same order in all files.
 * See test/RunBUFU.sh and test/testDTH.sh for example how to run with this module
 * Also see documentation in doc/README=DTH.md
*/

#include <filesystem>
#include <queue>
#include <regex>

#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Utilities/interface/DTHHeaders.h"

class FEDRawDataCollection;

class DataModeDTH : public DataMode {
public:
  DataModeDTH(DAQSource* daqSource, bool verifyChecksum) : DataMode(daqSource), verifyChecksum_(verifyChecksum) {}
  ~DataModeDTH() override {}
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  //non-virtual
  edm::Timestamp fillFEDRawDataCollection(FEDRawDataCollection& rawData);

  int dataVersion() const override { return detectedDTHversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedDTHversion_ = 1;  //TODO: read version
  }

  uint32_t headerSize() const override { return sizeof(evf::DTHOrbitHeader_v1); }

  bool versionCheck() const override { return detectedDTHversion_ == 1; }

  uint64_t dataBlockSize() const override { return dataBlockSize_; }

  void makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) override;

  bool nextEventView(RawInputFile*) override;
  bool blockChecksumValid() override { return checksumValid_; }
  bool checksumValid() override { return checksumValid_; }
  std::string getChecksumError() const override { return checksumError_; }

  bool isRealData() const { return true; }  //this flag could be added to RU/BU-generated index

  uint32_t run() const override { return firstOrbitHeader_->runNumber(); }

  bool dataBlockCompleted() const override { return blockCompleted_; }

  bool requireHeader() const override { return false; }

  bool fitToBuffer() const override { return true; }

  bool isMultiDir() const override { return true; }

  void unpackFile(RawInputFile*) override {}

  bool dataBlockInitialized() const override { return dataBlockInitialized_; }

  void setDataBlockInitialized(bool val) override { dataBlockInitialized_ = val; }

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override {}

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                            std::vector<int> const& numSources,
                            std::vector<int> const& sourceIDs,
                            std::string const& sourceIdentifier,
                            std::string const& runDir) override;

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName,
                                                                  bool fileListMode) const override;

  bool hasEventCounterCallback() const override { return true; }
  int eventCounterCallback(std::string const& name, int& fd, int64_t& fsize, uint32_t sLS, bool& found) const override;

private:
  bool verifyChecksum_;
  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  std::vector<std::filesystem::path> buPaths_;
  std::vector<int> buNumSources_;
  std::vector<std::string> buSourceStrings_;
  std::regex sid_pattern_;
  uint16_t detectedDTHversion_ = 0;
  evf::DTHOrbitHeader_v1* firstOrbitHeader_ = nullptr;
  uint64_t nextEventID_ = 0;
  std::vector<evf::DTHFragmentTrailer_v1*> eventFragments_;  //events in block (DTH trailer)
  //numFiles_ = 0;
  bool dataBlockInitialized_ = false;
  bool blockCompleted_ = true;

  std::vector<uint8_t*> addrsStart_;  //start of orbit payloads per source
  std::vector<uint8_t*> addrsEnd_;    //dth trailers per source (go through events from the end)

  bool checksumValid_ = false;
  std::string checksumError_;
  //total
  size_t dataBlockSize_ = 0;
  bool eventCached_ = false;
};

#endif  // EventFilter_Utilities_DAQSourceModelsDTH_h
