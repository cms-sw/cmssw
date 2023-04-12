#ifndef EventFilter_Utilities_DAQSourceModels_h
#define EventFilter_Utilities_DAQSourceModels_h

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"

//import InputChunk
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"

class DAQSource;

//evf?
class DataMode {
public:
  DataMode(DAQSource* daqSource) : daqSource_(daqSource) {}
  virtual ~DataMode() = default;
  virtual std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& makeDaqProvenanceHelpers() = 0;
  virtual void readEvent(edm::EventPrincipal& eventPrincipal) = 0;
  virtual int dataVersion() const = 0;
  virtual void detectVersion(unsigned char* fileBuf, uint32_t fileHeaderOffset) = 0;
  virtual uint32_t headerSize() const = 0;
  virtual bool versionCheck() const = 0;
  virtual uint64_t dataBlockSize() const = 0;
  virtual void makeDataBlockView(unsigned char* addr,
                                 size_t maxSize,
                                 std::vector<uint64_t> const& fileSizes,
                                 size_t fileHeaderSize) = 0;
  virtual bool nextEventView() = 0;
  virtual bool checksumValid() = 0;
  virtual std::string getChecksumError() const = 0;
  virtual bool isRealData() const = 0;
  virtual uint32_t run() const = 0;
  virtual bool dataBlockCompleted() const = 0;
  virtual bool requireHeader() const = 0;
  virtual bool fitToBuffer() const = 0;

  virtual bool dataBlockInitialized() const = 0;
  virtual void setDataBlockInitialized(bool) = 0;

  virtual void setTCDSSearchRange(uint16_t, uint16_t) = 0;
  virtual std::pair<bool, std::vector<std::string>> defineAdditionalFiles(std::string const& primaryName,
                                                                          bool fileListMode) const = 0;

  virtual bool isMultiDir() { return false; }
  virtual void makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) = 0;
  void setTesting(bool testing) { testing_ = testing; }

protected:
  DAQSource* daqSource_;
  bool testing_ = false;
};

#endif  // EventFilter_Utilities_DAQSourceModels_h
