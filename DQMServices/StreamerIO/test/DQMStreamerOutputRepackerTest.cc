#include <iomanip>
#include <filesystem>
#include <memory>
#include <sstream>

#include <zlib.h>
#include <fmt/printf.h>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

namespace dqmservices {

  class DQMStreamerOutputRepackerTest : public edm::StreamerOutputModuleBase {
  public:
    explicit DQMStreamerOutputRepackerTest(edm::ParameterSet const& ps);
    ~DQMStreamerOutputRepackerTest() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void start() override;
    void stop() override;
    void doOutputHeader(InitMsgBuilder const& init_message) override;
    void doOutputEvent(EventMsgBuilder const& msg) override;

    void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override{};
    void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override{};

  private:
    void openFile_(uint32_t run, uint32_t lumi);
    void closeFile();

  private:
    std::string streamLabel_;
    std::string outputPath_;

    std::unique_ptr<uint8_t[]> init_message_cache_;
    std::unique_ptr<StreamerOutputFile> streamFile_;
    uint32_t streamRun_;
    uint32_t streamLumi_;

    uint32_t eventsProcessedFile_;
    uint32_t eventsProcessedTotal_;

    std::string currentFileBase_;
    std::string currentFilePath_;
    std::string currentJsonPath_;
  };  // end-of-class-def

  DQMStreamerOutputRepackerTest::DQMStreamerOutputRepackerTest(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase::OutputModuleBase(ps), edm::StreamerOutputModuleBase(ps) {
    outputPath_ = ps.getUntrackedParameter<std::string>("outputPath");
    streamLabel_ = ps.getUntrackedParameter<std::string>("streamLabel");

    eventsProcessedTotal_ = 0;
    eventsProcessedFile_ = 0;
  }

  DQMStreamerOutputRepackerTest::~DQMStreamerOutputRepackerTest() {}

  void DQMStreamerOutputRepackerTest::openFile_(uint32_t run, uint32_t lumi) {
    if (streamFile_) {
      closeFile();
    }

    eventsProcessedFile_ = 0;

    currentFileBase_ = fmt::sprintf("run%06d_ls%04d_stream%s_local", run, lumi, streamLabel_);

    std::filesystem::path p = outputPath_;
    p /= fmt::sprintf("run%06d", run);

    std::filesystem::create_directories(p);

    currentFilePath_ = (p / currentFileBase_).string() + ".dat";
    currentJsonPath_ = (p / currentFileBase_).string() + ".jsn";

    edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing file: " << currentFilePath_;

    streamFile_.reset(new StreamerOutputFile(currentFilePath_));
    streamRun_ = run;
    streamLumi_ = lumi;

    if (init_message_cache_) {
      InitMsgView iview(init_message_cache_.get());
      streamFile_->write(iview);
    } else {
      edm::LogWarning("DQMStreamerOutputRepackerTest") << "Open file called before init message.";
    }
  }

  void DQMStreamerOutputRepackerTest::closeFile() {
    edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing json: " << currentJsonPath_;
    size_t fsize = std::filesystem::file_size(currentFilePath_);

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

    streamFile_.reset();
  }

  void DQMStreamerOutputRepackerTest::start() {}

  void DQMStreamerOutputRepackerTest::stop() { closeFile(); }

  void DQMStreamerOutputRepackerTest::doOutputHeader(InitMsgBuilder const& init_message_bldr) {
    edm::LogWarning("DQMStreamerOutputRepackerTest") << "doOutputHeader() method, initializing streams.";

    uint8_t* x = new uint8_t[init_message_bldr.size()];
    std::memcpy(x, init_message_bldr.startAddress(), init_message_bldr.size());
    init_message_cache_.reset(x);
  }

  void DQMStreamerOutputRepackerTest::doOutputEvent(EventMsgBuilder const& msg_bldr) {
    EventMsgView view(msg_bldr.startAddress());

    auto run = view.run();
    auto lumi = view.lumi();

    if ((!streamFile_) || (streamRun_ != run) || (streamLumi_ != lumi)) {
      openFile_(run, lumi);
    }

    eventsProcessedFile_ += 1;
    eventsProcessedTotal_ += 1;
    edm::LogAbsolute("DQMStreamerOutputRepackerTest") << "Writing event.";

    streamFile_->write(view);
  }

  void DQMStreamerOutputRepackerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::StreamerOutputModuleBase::fillDescription(desc);

    desc.addUntracked<std::string>("outputPath", "./output/")->setComment("File output path.");

    desc.addUntracked<std::string>("streamLabel", "DQM")->setComment("Stream label used in json discovery.");

    descriptions.add("DQMStreamerOutputRepackerTest", desc);
  }

}  // namespace dqmservices

#include "FWCore/Framework/interface/MakerMacros.h"

typedef dqmservices::DQMStreamerOutputRepackerTest DQMStreamerOutputRepackerTest;
DEFINE_FWK_MODULE(DQMStreamerOutputRepackerTest);
