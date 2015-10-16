#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
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

class DQMStreamerOutputModule : public edm::StreamerOutputModuleBase {
 public:
  explicit DQMStreamerOutputModule(edm::ParameterSet const& ps);
  virtual ~DQMStreamerOutputModule();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void start() const;
  virtual void stop() const;

  // no clue why the hell these are const in the parent class
  // so these can't be used - I will do some very bad mutable magic
  virtual void doOutputHeader(InitMsgBuilder const& init_message) const;
  virtual void doOutputEvent(EventMsgBuilder const& msg) const;

  virtual void beginLuminosityBlock(edm::LuminosityBlockPrincipal const&,
                                    edm::ModuleCallingContext const*) override;

  virtual void endLuminosityBlock(edm::LuminosityBlockPrincipal const&,
                                  edm::ModuleCallingContext const*) override;

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

DQMStreamerOutputModule::DQMStreamerOutputModule(edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::StreamerOutputModuleBase(ps),
      streamLabel_(ps.getUntrackedParameter<std::string>("streamLabel")),
      runInputDir_(ps.getUntrackedParameter<std::string>("runInputDir", "")),
      processed_(0),
      flagLumiRemap_(true),
      currentLumi_(0),
      currentRun_(0) {

  edm::LogInfo("DQMStreamerOutputModule") << "Writing .dat files to "
                                          << runInputDir_;

  if (!boost::filesystem::is_directory(runInputDir_)) {
    std::cout << "<open> FU dir not found. Creating..." << std::endl;
    boost::filesystem::create_directories(runInputDir_);
  }
}

DQMStreamerOutputModule::~DQMStreamerOutputModule() {}

void DQMStreamerOutputModule::doOutputHeader(InitMsgBuilder const& i) const {

  init_message_cache_.reset(new InitMsgView(i.startAddress()));
}

void DQMStreamerOutputModule::doOutputEvent(EventMsgBuilder const& msg) const {

  ++processed_;

  EventMsgView eview(msg.startAddress());
  stream_writer_events_->write(eview);
  // You can't use msg in DQMStreamerOutputModule after this point
}

void DQMStreamerOutputModule::beginLuminosityBlock(
    edm::LuminosityBlockPrincipal const& ls, edm::ModuleCallingContext const*) {

  std::cout << "DQMStreamerOutputModule : begin lumi." << std::endl;

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
    std::cout << "DQMStreamerOutputModule : creating run directory: " << p
              << std::endl;
    boost::filesystem::create_directories(p);
  }

  stream_writer_events_.reset(new StreamerOutputFile(path));
  eventsFile_ = path;

  std::cout << "DQMStreamerOutputModule : writing init message." << std::endl;
  stream_writer_events_->write(*init_message_cache_);

  processed_ = 0;
}

void DQMStreamerOutputModule::endLuminosityBlock(
    edm::LuminosityBlockPrincipal const& ls, edm::ModuleCallingContext const*) {

  std::cout << "DQMStreamerOutputModule : end lumi " << std::endl;
  stream_writer_events_.reset();

  // output jsn file
  std::string path =
      str(boost::format("%s/run%06d/run%06d_ls%04d%s.jsn") % runInputDir_ %
          currentRun_ % currentRun_ % currentLumi_ % streamLabel_);
  std::cout << "DQMStreamerOutputModule : writing json: " << path << std::endl;

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

void DQMStreamerOutputModule::start() const {}

void DQMStreamerOutputModule::stop() const {
  std::cout << "DQMStreamerOutputModule : end run" << std::endl;
  stream_writer_events_.reset();

  // output jsn file
  std::string path = str(boost::format("%s/run%06d/run%06d_ls%04d_EoR.jsn") %
                         runInputDir_ % currentRun_ % currentRun_ % 0);
  std::cout << "DQMStreamerOutputModule : writing json: " << path << std::endl;

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

void DQMStreamerOutputModule::fillDescriptions(
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

typedef edm::DQMStreamerOutputModule DQMStreamerOutputModule;
DEFINE_FWK_MODULE(DQMStreamerOutputModule);
