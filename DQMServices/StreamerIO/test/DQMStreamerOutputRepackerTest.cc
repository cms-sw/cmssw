#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

#include <zlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iomanip>
#include <memory>
#include <sstream>

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerOutputFile.h"

#include "DQMServices/Components/src/DQMFileSaver.h"

namespace dqmservices {

class DQMStreamerOutputRepackerTest : public edm::StreamerOutputModuleBase {
 public:
  explicit DQMStreamerOutputRepackerTest(edm::ParameterSet const& ps);
  virtual ~DQMStreamerOutputRepackerTest();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void start() override;
  virtual void stop() override;
  virtual void doOutputHeader(InitMsgBuilder const& init_message) override;
  virtual void doOutputEvent(EventMsgBuilder const& msg) override;

  virtual void beginLuminosityBlock(
      edm::LuminosityBlockForOutput const&) override{};
  virtual void endLuminosityBlock(
      edm::LuminosityBlockForOutput const&) override{};

 private:
  void openFile();
  void closeFile();

 private:
  std::string streamLabel_;
  std::string outputPath_;

  unsigned int runNumber_;
  unsigned int eventsPerFile_;

  std::unique_ptr<StreamerOutputFile> stream_writer_events_;
  std::unique_ptr<uint8_t[]> init_message_cache_;

  long eventsProcessedTotal_;
  long eventsProcessedFile_;
  int currentFileIndex_;

  std::string currentFileBase_;

  std::string currentFilePath_;
  std::string currentJsonPath_;
};  // end-of-class-def

DQMStreamerOutputRepackerTest::DQMStreamerOutputRepackerTest(
    edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::StreamerOutputModuleBase(ps) {
  outputPath_ = ps.getUntrackedParameter<std::string>("outputPath");
  streamLabel_ = ps.getUntrackedParameter<std::string>("streamLabel");
  runNumber_ = ps.getUntrackedParameter<unsigned int>("runNumber");
  eventsPerFile_ = ps.getUntrackedParameter<unsigned int>("eventsPerFile");

  eventsProcessedTotal_ = 0;
  eventsProcessedFile_ = 0;
  currentFileIndex_ = 0;
}

DQMStreamerOutputRepackerTest::~DQMStreamerOutputRepackerTest() {}

void DQMStreamerOutputRepackerTest::openFile() {
  if (stream_writer_events_) {
    closeFile();
  }

  currentFileIndex_ += 1;
  eventsProcessedFile_ = 0;

  currentFileBase_ = str(boost::format("run%06d_ls%04d_stream%s_local") %
                         runNumber_ % currentFileIndex_ % streamLabel_);

  boost::filesystem::path p = outputPath_;
  p /= str(boost::format("run%06d") % runNumber_);

  boost::filesystem::create_directories(p);

  currentFilePath_ = (p / currentFileBase_).string() + ".dat";
  currentJsonPath_ = (p / currentFileBase_).string() + ".jsn";

  edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing file: "
                                                << currentFilePath_;

  stream_writer_events_.reset(new StreamerOutputFile(currentFilePath_));

  if (init_message_cache_) {
    InitMsgView iview(init_message_cache_.get());
    stream_writer_events_->write(iview);
  } else {
    edm::LogWarning("DQMStreamerOutputRepackerTest")
        << "Open file called before init message.";
  }
}

void DQMStreamerOutputRepackerTest::closeFile() {
  edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing json: "
                                                << currentJsonPath_;
  size_t fsize = boost::filesystem::file_size(currentFilePath_);

  using namespace boost::property_tree;
  ptree pt;
  ptree data;

  ptree child1, child2, child3, child4, child5;
  child1.put("", eventsProcessedTotal_);  // Processed
  child2.put("", eventsProcessedFile_);   // Accepted
  child3.put("", 0);                      // Errors
  child4.put("", currentFileBase_ + ".dat");
  child5.put("", fsize);

  data.push_back(std::make_pair("", child1));
  data.push_back(std::make_pair("", child2));
  data.push_back(std::make_pair("", child3));
  data.push_back(std::make_pair("", child4));
  data.push_back(std::make_pair("", child5));

  pt.add_child("data", data);
  pt.put("definition", "");
  pt.put("source", "");

  std::string json_tmp = currentJsonPath_ + ".open";
  write_json(json_tmp, pt);
  ::rename(json_tmp.c_str(), currentJsonPath_.c_str());

  stream_writer_events_.reset();
}

void DQMStreamerOutputRepackerTest::start() {}

void DQMStreamerOutputRepackerTest::stop() { closeFile(); }

void DQMStreamerOutputRepackerTest::doOutputHeader(
    InitMsgBuilder const& init_message_bldr) {
  edm::LogWarning("DQMStreamerOutputRepackerTest")
      << "doOutputHeader() method, initializing streams.";

  uint8_t* x = new uint8_t[init_message_bldr.size()];
  std::memcpy(x, init_message_bldr.startAddress(), init_message_bldr.size());
  init_message_cache_.reset(x);
  openFile();
}

void DQMStreamerOutputRepackerTest::doOutputEvent(EventMsgBuilder const& msg_bldr) {
  if (eventsProcessedFile_ >= eventsPerFile_) {
    openFile();
  }

  eventsProcessedTotal_ += 1;
  eventsProcessedFile_ += 1;

  edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing event: "
                                                << eventsProcessedTotal_;

  EventMsgView view(msg_bldr.startAddress());
  stream_writer_events_->write(view);
}

void DQMStreamerOutputRepackerTest::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  edm::StreamerOutputModuleBase::fillDescription(desc);

  desc.addUntracked<std::string>("outputPath")->setComment("File output path.");

  desc.addUntracked<std::string>("streamLabel")
      ->setComment("Stream label used in json discovery.");

  desc.addUntracked<unsigned int>("runNumber")
      ->setComment("Run number passed via configuration file.");

  desc.addUntracked<unsigned int>("eventsPerFile")
      ->setComment("Number of events per file.");

  descriptions.add("DQMStreamerOutputRepackerTest", desc);
}

}  // end of namespace

#include "EventFilter/Utilities/plugins/RecoEventWriterForFU.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef dqmservices::DQMStreamerOutputRepackerTest DQMStreamerOutputRepackerTest;
DEFINE_FWK_MODULE(DQMStreamerOutputRepackerTest);
