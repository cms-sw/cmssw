#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"

namespace edm {

class DQMStreamerOutputModuleTest : public edm::StreamerOutputModuleBase {
 public:
  explicit DQMStreamerOutputModuleTest(edm::ParameterSet const& ps);
  virtual ~DQMStreamerOutputModuleTest();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void start() override;
  virtual void stop() override;

  // no clue why the hell these are const in the parent class
  // so these can't be used - I will do some very bad mutable magic
  //
  // NOTE: these are no longer const in the parent class, (or here).
  // So perhaps they can be used now?
  //
  virtual void doOutputHeader(InitMsgBuilder const& init_message) override;
  virtual void doOutputEvent(EventMsgBuilder const& msg) override;

  virtual void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

  virtual void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

 private:
  std::string streamLabel_;
  std::string eventsFile_;
  std::string runInputDir_;

  mutable int processed_;
  bool flagLumiRemap_;
  int currentLumi_;
  int currentRun_;

  mutable boost::shared_ptr<StreamerOutputFile> stream_writer_events_;
  mutable boost::shared_ptr<InitMsgView> init_message_cache_;
};  //end-of-class-def

DQMStreamerOutputModuleTest::DQMStreamerOutputModuleTest(edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::StreamerOutputModuleBase(ps),
      streamLabel_(ps.getUntrackedParameter<std::string>("streamLabel")),
      runInputDir_(ps.getUntrackedParameter<std::string>("runInputDir", "")),
      processed_(0),
      flagLumiRemap_(true),
      currentLumi_(0),
      currentRun_(0) {

  edm::LogInfo("DQMStreamerOutputModuleTest") << "Writing .dat files to "
                                          << runInputDir_;

  if (!boost::filesystem::is_directory(runInputDir_)) {
    std::cout << "<open> FU dir not found. Creating..." << std::endl;
    boost::filesystem::create_directories(runInputDir_);
  }
}

DQMStreamerOutputModuleTest::~DQMStreamerOutputModuleTest() {}

void DQMStreamerOutputModuleTest::doOutputHeader(InitMsgBuilder const& i) {

  init_message_cache_.reset(new InitMsgView(i.startAddress()));
}

void DQMStreamerOutputModuleTest::doOutputEvent(EventMsgBuilder const& msg) {

  ++processed_;

  EventMsgView eview(msg.startAddress());
  stream_writer_events_->write(eview);
  // You can't use msg in DQMStreamerOutputModuleTest after this point
}

void DQMStreamerOutputModuleTest::beginLuminosityBlock(
    edm::LuminosityBlockForOutput const& ls) {

  std::cout << "DQMStreamerOutputModuleTest : begin lumi." << std::endl;

  if (flagLumiRemap_) {
    currentLumi_++;
  } else {
    currentLumi_ = ls.luminosityBlock();
  }

  currentRun_ = ls.run();

  std::string path =
      str(boost::format("%s/run%06d/run%06d_ls%04d%s.dat") % runInputDir_ %
          currentRun_ % currentRun_ % currentLumi_ % streamLabel_);

  boost::filesystem::path p(path);
  p = p.parent_path();

  if (!boost::filesystem::is_directory(p)) {
    std::cout << "DQMStreamerOutputModuleTest : creating run directory: " << p
              << std::endl;
    boost::filesystem::create_directories(p);
  }

  stream_writer_events_.reset(new StreamerOutputFile(path));
  eventsFile_ = path;

  std::cout << "DQMStreamerOutputModuleTest : writing init message." << std::endl;
  stream_writer_events_->write(*init_message_cache_);

  processed_ = 0;
}

void DQMStreamerOutputModuleTest::endLuminosityBlock(
    edm::LuminosityBlockForOutput const& ls) {

  std::cout << "DQMStreamerOutputModuleTest : end lumi " << std::endl;
  stream_writer_events_.reset();

  // output jsn file
  std::string path =
      str(boost::format("%s/run%06d/run%06d_ls%04d%s.jsn") % runInputDir_ %
          currentRun_ % currentRun_ % currentLumi_ % streamLabel_);
  std::cout << "DQMStreamerOutputModuleTest : writing json: " << path << std::endl;

  using namespace boost::property_tree;
  ptree pt;
  ptree data;

  ptree child1, child2, child3;

  child1.put("", processed_);   // Processed
  child2.put("", processed_);   // Accepted
  child3.put("", eventsFile_);  // filelist

  data.push_back(std::make_pair("", child1));
  data.push_back(std::make_pair("", child2));
  data.push_back(std::make_pair("", child3));

  pt.add_child("data", data);
  pt.put("definition", "/non-existant/");
  pt.put("source", "--hostname--");

  std::ofstream file(path);
  write_json(file, pt, true);
  file.close();
}

void DQMStreamerOutputModuleTest::start() {}

void DQMStreamerOutputModuleTest::stop() {
  std::cout << "DQMStreamerOutputModuleTest : end run" << std::endl;
  stream_writer_events_.reset();

  // output jsn file
  std::string path = str(boost::format("%s/run%06d/run%06d_ls%04d_EoR.jsn") %
                         runInputDir_ % currentRun_ % currentRun_ % 0);
  std::cout << "DQMStreamerOutputModuleTest : writing json: " << path << std::endl;

  using namespace boost::property_tree;
  ptree pt;
  ptree data;

  ptree child1, child2, child3, child4;

  child1.put("", processed_);    // Processed
  child2.put("", processed_);    // Processed
  child3.put("", processed_);    // Accepted
  child4.put("", currentLumi_);  // number of lumi

  data.push_back(std::make_pair("", child1));
  data.push_back(std::make_pair("", child2));
  data.push_back(std::make_pair("", child3));
  data.push_back(std::make_pair("", child4));

  pt.add_child("data", data);
  pt.put("definition", "/non-existant/");
  pt.put("source", "--hostname--");

  std::ofstream file(path);
  write_json(file, pt, true);
  file.close();
}

void DQMStreamerOutputModuleTest::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  edm::StreamerOutputModuleBase::fillDescription(desc);

  desc.addUntracked<std::string>("runInputDir")
      ->setComment("Top level output directory");

  desc.addUntracked<std::string>("streamLabel")
      ->setComment("Stream label.");

  descriptions.add("streamerOutput", desc);
}

}  // end of namespace-edm

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/Utilities/plugins/RecoEventWriterForFU.h"

typedef edm::DQMStreamerOutputModuleTest DQMStreamerOutputModuleTest;
DEFINE_FWK_MODULE(DQMStreamerOutputModuleTest);
