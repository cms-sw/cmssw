#include "DQM/L1TMonitorClient/interface/L1TdeStage2RegionalShowerClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TdeStage2RegionalShowerClient::L1TdeStage2RegionalShowerClient(const edm::ParameterSet &ps)
    : monitorDir_(ps.getUntrackedParameter<string>("monitorDir")) {}

L1TdeStage2RegionalShowerClient::~L1TdeStage2RegionalShowerClient() {}

void L1TdeStage2RegionalShowerClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                                            DQMStore::IGetter &igetter,
                                                            const edm::LuminosityBlock &lumiSeg,
                                                            const edm::EventSetup &c) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TdeStage2RegionalShowerClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TdeStage2RegionalShowerClient::book(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorDir_);

  emtfShowerDataSummary_eff_ = iBooker.book2D(
      "emtf_shower_data_summary_eff", "Efficiency of data EMTF shower being correctly emulated", 6, 1, 7, 4, 0, 4);
  emtfShowerEmulSummary_eff_ = iBooker.book2D(
      "emtf_shower_emul_summary_eff", "Fraction of emulated EMTF shower without matching data shower", 6, 1, 7, 4, 0, 4);

  // x labels
  emtfShowerDataSummary_eff_->setAxisTitle("Chamber", 1);
  emtfShowerEmulSummary_eff_->setAxisTitle("Chamber", 1);

  // plotting option
  emtfShowerDataSummary_eff_->setOption("colz");
  emtfShowerEmulSummary_eff_->setOption("colz");

  // y labels
  emtfShowerDataSummary_eff_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerDataSummary_eff_->setBinLabel(2, "ME- Nom", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(2, "ME- Nom", 2);
  emtfShowerDataSummary_eff_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerDataSummary_eff_->setBinLabel(4, "ME+ Tight", 2);
  emtfShowerEmulSummary_eff_->setBinLabel(4, "ME+ Tight", 2);
}

void L1TdeStage2RegionalShowerClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *emtfShowerDataSummary_denom_ = igetter.get(monitorDir_ + "/emtf_shower_data_summary_denom");
  MonitorElement *emtfShowerDataSummary_num_ = igetter.get(monitorDir_ + "/emtf_shower_data_summary_num");

  MonitorElement *emtfShowerEmulSummary_denom_ = igetter.get(monitorDir_ + "/emtf_shower_emul_summary_denom");
  MonitorElement *emtfShowerEmulSummary_num_ = igetter.get(monitorDir_ + "/emtf_shower_emul_summary_num");

  if (emtfShowerDataSummary_denom_ == nullptr or emtfShowerDataSummary_num_ == nullptr or
      emtfShowerEmulSummary_denom_ == nullptr or emtfShowerEmulSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeStage2RegionalShowerClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary histograms for the harvesting";
    return;
  }

  emtfShowerDataSummary_eff_->getTH2F()->Divide(
      emtfShowerDataSummary_num_->getTH2F(), emtfShowerDataSummary_denom_->getTH2F(), 1, 1, "");
  emtfShowerEmulSummary_eff_->getTH2F()->Divide(
      emtfShowerEmulSummary_num_->getTH2F(), emtfShowerEmulSummary_denom_->getTH2F(), 1, 1, "");
}
