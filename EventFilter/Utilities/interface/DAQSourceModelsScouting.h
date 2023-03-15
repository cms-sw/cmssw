#ifndef EventFilter_Utilities_DAQSourceModelsScouting_h
#define EventFilter_Utilities_DAQSourceModelsScouting_h

#include <memory>

#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace scouting {
  struct muon {
    uint32_t f;
    uint32_t s;
  };

  struct block {
    uint32_t bx;
    uint32_t orbit;
    muon mu[16];
  };

  struct masks {
    static constexpr uint32_t phiext = 0x3ff;
    static constexpr uint32_t pt = 0x1ff;
    static constexpr uint32_t qual = 0xf;
    static constexpr uint32_t etaext = 0x1ff;
    static constexpr uint32_t etaextv = 0xff;
    static constexpr uint32_t etaexts = 0x100;
    static constexpr uint32_t iso = 0x3;
    static constexpr uint32_t chrg = 0x1;
    static constexpr uint32_t chrgv = 0x1;
    static constexpr uint32_t index = 0x7f;
    static constexpr uint32_t phi = 0x3ff;
    static constexpr uint32_t eta = 0x1ff;
    static constexpr uint32_t etav = 0xff;
    static constexpr uint32_t etas = 0x100;
    static constexpr uint32_t phiv = 0x1ff;
    static constexpr uint32_t phis = 0x200;
    static constexpr uint32_t sv = 0x3;
  };

  struct shifts {
    static constexpr uint32_t phiext = 0;
    static constexpr uint32_t pt = 10;
    static constexpr uint32_t qual = 19;
    static constexpr uint32_t etaext = 23;
    static constexpr uint32_t iso = 0;
    static constexpr uint32_t chrg = 2;
    static constexpr uint32_t chrgv = 3;
    static constexpr uint32_t index = 4;
    static constexpr uint32_t phi = 11;
    static constexpr uint32_t eta = 21;
    static constexpr uint32_t rsv = 30;
  };

  struct gmt_scales {
    static constexpr float pt_scale = 0.5;
    static constexpr float phi_scale = 2. * M_PI / 576.;
    static constexpr float eta_scale = 0.0870 / 8;  //9th MS bit is sign
    static constexpr float phi_range = M_PI;
  };

  struct header_shifts {
    static constexpr uint32_t bxmatch = 24;
    static constexpr uint32_t mAcount = 16;
    static constexpr uint32_t orbitmatch = 8;
    static constexpr uint32_t mBcount = 0;
  };

  struct header_masks {
    static constexpr uint32_t bxmatch = 0xff << header_shifts::bxmatch;
    static constexpr uint32_t mAcount = 0xf << header_shifts::mAcount;
    static constexpr uint32_t orbitmatch = 0xff << header_shifts::orbitmatch;
    static constexpr uint32_t mBcount = 0xf;
  };

}  //namespace scouting

class DataModeScoutingRun2Muon : public DataMode {
public:
  DataModeScoutingRun2Muon(DAQSource* daqSource) : DataMode(daqSource) {
    dummyLVec_ = std::make_unique<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>();
  }

  ~DataModeScoutingRun2Muon() override{};

  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

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

  //true for scouting muon
  bool dataBlockCompleted() const override { return true; }

  bool requireHeader() const override { return true; }

  bool fitToBuffer() const override { return true; }

  bool dataBlockInitialized() const override { return true; }

  void setDataBlockInitialized(bool) override{};

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override { return; }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) override {}

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName, bool) const override {
    return std::make_pair(true, std::vector<std::string>());
  }

  char* readPayloadPos() { return (char*)event_->payload(); }

private:
  void unpackOrbit(BXVector<l1t::Muon>* muons, char* buf, size_t len);

  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  uint16_t detectedFRDversion_ = 0;
  size_t headerSize_ = 0;
  std::unique_ptr<FRDEventMsgView> event_;

  std::unique_ptr<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>> dummyLVec_;

  unsigned char* dataBlockAddr_ = nullptr;
  size_t dataBlockMax_ = 0;
  bool eventCached_ = false;
};

class DataModeScoutingRun2Multi : public DataMode {
public:
  DataModeScoutingRun2Multi(DAQSource* daqSource) : DataMode(daqSource) {
    dummyLVec_ = std::make_unique<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>();
  }

  ~DataModeScoutingRun2Multi() override{};

  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() override;
  void readEvent(edm::EventPrincipal& eventPrincipal) override;

  int dataVersion() const override { return detectedFRDversion_; }
  void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) override {
    detectedFRDversion_ = *((uint16_t*)(fileBuf + fileHeaderOffset));
  }

  uint32_t headerSize() const override { return FRDHeaderVersionSize[detectedFRDversion_]; }

  bool versionCheck() const override { return detectedFRDversion_ <= FRDHeaderMaxVersion; }

  uint64_t dataBlockSize() const override {
    //TODO: adjust to multiple objects
    return events_[0]->size();
  }

  void makeDataBlockView(unsigned char* addr,
                         size_t maxSize,
                         std::vector<uint64_t> const& fileSizes,
                         size_t fileHeaderSize) override {
    fileHeaderSize_ = fileHeaderSize;
    numFiles_ = fileSizes.size();
    //add offset address for each file payload
    startAddrs_.clear();
    startAddrs_.push_back(addr);
    dataBlockAddrs_.clear();
    dataBlockAddrs_.push_back(addr);
    dataBlockMaxAddrs_.clear();
    dataBlockMaxAddrs_.push_back(addr + fileSizes[0] - fileHeaderSize);
    auto fileAddr = addr;
    for (unsigned int i = 1; i < fileSizes.size(); i++) {
      fileAddr += fileSizes[i - 1];
      startAddrs_.push_back(fileAddr);
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

  //true for DAQ3 FRD
  bool dataBlockCompleted() const override { return blockCompleted_; }

  bool requireHeader() const override { return true; }

  bool dataBlockInitialized() const override { return dataBlockInitialized_; }

  void setDataBlockInitialized(bool val) override { dataBlockInitialized_ = val; };

  void setTCDSSearchRange(uint16_t MINTCDSuTCAFEDID, uint16_t MAXTCDSuTCAFEDID) override { return; }

  void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) override {
    //receive directory paths for multiple input files ('striped')
  }

  std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName,
                                                                  bool fileListMode) const override;

private:
  bool makeEvents();
  void unpackMuonOrbit(BXVector<l1t::Muon>* muons, char* buf, size_t len);

  std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>> daqProvenanceHelpers_;
  uint16_t detectedFRDversion_ = 0;
  size_t headerSize_ = 0;
  std::vector<std::unique_ptr<FRDEventMsgView>> events_;

  std::unique_ptr<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>> dummyLVec_;

  unsigned char* dataBlockAddr_ = nullptr;

  //debugging
  std::vector<unsigned char*> startAddrs_;
  std::vector<unsigned char*> dataBlockAddrs_;
  std::vector<unsigned char*> dataBlockMaxAddrs_;
  size_t dataBlockMax_ = 0;
  size_t fileHeaderSize_ = 0;
  short numFiles_ = 0;
  bool eventCached_ = false;
  bool dataBlockInitialized_ = false;
  bool blockCompleted_ = true;
};

#endif  // EventFilter_Utilities_DAQSourceModelsScouting_h
