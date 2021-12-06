#include "DQM/L1TMonitorClient/interface/L1TdeStage2ShowerClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TdeStage2ShowerClient::L1TdeStage2ShowerClient(const edm::ParameterSet &ps)
    : monitorDir_(ps.getUntrackedParameter<string>("monitorDir")) {}

L1TdeStage2ShowerClient::~L1TdeStage2ShowerClient() {}

void L1TdeStage2ShowerClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                              DQMStore::IGetter &igetter,
                                              const edm::LuminosityBlock &lumiSeg,
                                              const edm::EventSetup &c) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TdeStage2ShowerClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TdeStage2ShowerClient::book(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorDir_);

  emtfShowerDataSummary_eff_ = iBooker.book2D(
      "emtf_shower_data_summary_eff", "Efficiency of data EMTF shower being correctly emulated", 6, 1, 7, 2, 0, 2);
  emtfShowerEmulSummary_eff_ = iBooker.book2D(
      "emtf_shower_emul_summary_eff", "Fraction of emulated EMTF shower without matching data shower", 6, 1, 7, 2, 0, 2);

  // x labels
  emtfShowerDataSummary_eff_->setAxisTitle("Chamber", 1);
  emtfShowerEmulSummary_eff_->setAxisTitle("Chamber", 1);

  // plotting option
  emtfShowerDataSummary_eff_->setOption("colz");
  emtfShowerEmulSummary_eff_->setOption("colz");

  // y labels
  emtfShowerDataSummary_eff_->setBinLabel(1, "ME-", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(1, "ME-", 2);
  emtfShowerDataSummary_eff_->setBinLabel(2, "ME+", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(2, "ME+", 2);
}

void L1TdeStage2ShowerClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *emtfShowerDataSummary_denom_ = igetter.get(monitorDir_ + "/emtf_shower_data_summary_denom");
  MonitorElement *emtfShowerDataSummary_num_ = igetter.get(monitorDir_ + "/emtf_shower_data_summary_num");

  MonitorElement *emtfShowerEmulSummary_denom_ = igetter.get(monitorDir_ + "/emtf_shower_emul_summary_denom");
  MonitorElement *emtfShowerEmulSummary_num_ = igetter.get(monitorDir_ + "/emtf_shower_emul_summary_num");

  emtfShowerDataSummary_eff_->getTH2F()->Divide(emtfShowerDataSummary_num_->getTH2F(), emtfShowerDataSummary_denom_->getTH2F(), 1, 1, "");
  emtfShowerEmulSummary_eff_->getTH2F()->Divide(emtfShowerEmulSummary_num_->getTH2F(), emtfShowerEmulSummary_denom_->getTH2F(), 1, 1, "");
}
