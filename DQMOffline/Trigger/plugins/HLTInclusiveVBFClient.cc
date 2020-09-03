/*
 *  \author N. Srimanobhas
 */

#include "DQMOffline/Trigger/interface/HLTInclusiveVBFClient.h"

using namespace std;
using namespace edm;

HLTInclusiveVBFClient::HLTInclusiveVBFClient(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  //
  dbe_ = Service<DQMStore>().operator->();

  //
  if (!dbe_) {
    edm::LogError("HLTInclusiveVBFClient")
        << "unable to get DQMStore service, upshot is no client histograms will be made";
  }

  //
  debug_ = false;
  verbose_ = false;

  //
  processname_ = iConfig.getParameter<std::string>("processname");

  //
  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_)
    std::cout << hltTag_ << std::endl;

  //
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
  if (dbe_)
    dbe_->setCurrentFolder(dirName_);
}

HLTInclusiveVBFClient::~HLTInclusiveVBFClient() = default;

void HLTInclusiveVBFClient::beginRun(const edm::Run& r, const edm::EventSetup& context) {}

void HLTInclusiveVBFClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void HLTInclusiveVBFClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
  runClient_();
}

void HLTInclusiveVBFClient::endRun(const edm::Run& r, const edm::EventSetup& context) {}

void HLTInclusiveVBFClient::runClient_() {
  if (!dbe_)
    return;  //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);

  LogDebug("HLTInclusiveVBFClient") << "runClient" << std::endl;
  if (debug_)
    std::cout << "runClient" << std::endl;
}
