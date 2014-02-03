#ifndef EventFilter_Utilities_FedRawDataInputSource_h
#define EventFilter_Utilities_FedRawDataInputSource_h

#include <memory>
#include <stdio.h>

#include "boost/filesystem.hpp"

#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

class FEDRawDataCollection;
class InputSourceDescription;
class ParameterSet;

namespace evf {
class FastMonitoringService;
}

namespace jsoncollector {
class DataPointDefinition;
}

class FedRawDataInputSource: public edm::RawInputSource {

public:
  explicit FedRawDataInputSource(edm::ParameterSet const&,edm::InputSourceDescription const&);
  virtual ~FedRawDataInputSource();

protected:
  virtual bool checkNextEvent() override;
  virtual void read(edm::EventPrincipal& eventPrincipal) override;

private:
  virtual void preForkReleaseResources() override;
  virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>) override;
  virtual void rewind_() override;

  void maybeOpenNewLumiSection(const uint32_t lumiSection);
  int cacheNextEvent();
  edm::Timestamp fillFEDRawDataCollection(std::auto_ptr<FEDRawDataCollection>&) const;
  void closeCurrentFile();
  int openNextFile();
  int searchForNextFile();
  bool grabNextJsonFile(boost::filesystem::path const&);
  void openDataFile(std::string const&);
  bool eofReached() const;
  int readNextChunkIntoBuffer();
  void renameToNextFree() const;

  std::string defPath_;

  unsigned int eventChunkSize_; // for buffered read-ahead
  unsigned int eventChunkBlock_; // how much read(2) asks at the time
  unsigned int readBlocks_;

  // get LS from filename instead of event header
  const bool getLSFromFilename_;
  const bool verifyAdler32_;
  const bool testModeNoBuilderUnit_;
  
  const edm::RunNumber_t runNumber_;

  const std::string buInputDir_;
  const std::string fuOutputDir_;

  const edm::DaqProvenanceHelper daqProvenanceHelper_;

  std::unique_ptr<FRDEventMsgView> event_;

  boost::filesystem::path openFile_;
  edm::EventID eventID_;

  unsigned int currentLumiSection_;
  boost::filesystem::path currentInputJson_;
  unsigned int currentInputEventCount_;

  bool eorFileSeen_;

  unsigned char *dataBuffer_; // temporarily hold multiple event data
  unsigned char *bufferCursor_;
  uint32_t bufferLeft_;

  jsoncollector::DataPointDefinition *dpd_;

  unsigned int eventsThisLumi_;
  evf::FastMonitoringService *fms_ = nullptr;

  bool fileIsOver_ = true;
  int fileDescriptor_ = -1;

};

#endif // EventFilter_Utilities_FedRawDataInputSource_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
