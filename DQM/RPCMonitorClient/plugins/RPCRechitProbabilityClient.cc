// Original Author:  Anna Cimmino
#include "DQM/RPCMonitorClient/interface/RPCRecHitProbabilityClient.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <fmt/format.h>

RPCRecHitProbabilityClient::RPCRecHitProbabilityClient(const edm::ParameterSet &iConfig) {
  edm::LogVerbatim("rpcdqmclient") << "[RPCRecHitProbabilityClient]: Constructor";

  const std::string subsystemFolder = iConfig.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  const std::string recHitTypeFolder = iConfig.getUntrackedParameter<std::string>("MuonFolder", "Muon");

  const std::string summaryFolder = iConfig.getUntrackedParameter<std::string>("GlobalFolder", "SummaryHistograms");

  globalFolder_ = subsystemFolder + "/" + recHitTypeFolder + "/" + summaryFolder;
}

void RPCRecHitProbabilityClient::beginJob() {
  edm::LogVerbatim("rpcrechitprobabilityclient") << "[RPCRecHitProbabilityClient]: Begin Job";
}

void RPCRecHitProbabilityClient::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                                       DQMStore::IGetter &,
                                                       edm::LuminosityBlock const &,
                                                       edm::EventSetup const &) {}

void RPCRecHitProbabilityClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  edm::LogVerbatim("rpcrechitprobabilityclient") << "[RPCRecHitProbabilityClient]: End Run";

  MonitorElement *NumberOfMuonEta = igetter.get(globalFolder_ + "/NumberOfMuonEta");
  MonitorElement *NumberOfMuonPt_B = igetter.get(globalFolder_ + "/NumberOfMuonPt_Barrel");
  MonitorElement *NumberOfMuonPt_EP = igetter.get(globalFolder_ + "/NumberOfMuonPt_EndcapP");
  MonitorElement *NumberOfMuonPt_EM = igetter.get(globalFolder_ + "/NumberOfMuonPt_EndcapM");
  MonitorElement *NumberOfMuonPhi_B = igetter.get(globalFolder_ + "/NumberOfMuonPhi_Barrel");
  MonitorElement *NumberOfMuonPhi_EP = igetter.get(globalFolder_ + "/NumberOfMuonPhi_EndcapP");
  MonitorElement *NumberOfMuonPhi_EM = igetter.get(globalFolder_ + "/NumberOfMuonPhi_EndcapM");

  if (NumberOfMuonEta == nullptr || NumberOfMuonPt_B == nullptr || NumberOfMuonPt_EP == nullptr ||
      NumberOfMuonPt_EM == nullptr || NumberOfMuonPhi_B == nullptr || NumberOfMuonPhi_EP == nullptr ||
      NumberOfMuonPhi_EM == nullptr)
    return;

  TH1F *NumberOfMuonEtaTH1F = NumberOfMuonEta->getTH1F();
  TH1F *NumberOfMuonPtBTH1F = NumberOfMuonPt_B->getTH1F();
  TH1F *NumberOfMuonPtEPTH1F = NumberOfMuonPt_EP->getTH1F();
  TH1F *NumberOfMuonPtEMTH1F = NumberOfMuonPt_EM->getTH1F();
  TH1F *NumberOfMuonPhiBTH1F = NumberOfMuonPhi_B->getTH1F();
  TH1F *NumberOfMuonPhiEPTH1F = NumberOfMuonPhi_EP->getTH1F();
  TH1F *NumberOfMuonPhiEMTH1F = NumberOfMuonPhi_EM->getTH1F();

  MonitorElement *recHit;
  for (int i = 1; i <= 6; i++) {
    recHit = igetter.get(fmt::format("{}/{}RecHitMuonEta", globalFolder_, i));
    if (recHit)
      recHit->getTH1F()->Divide(NumberOfMuonEtaTH1F);

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPtB", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPtBTH1F);
    }

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPhiB", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPhiBTH1F);
    }

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPtEP", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPtEPTH1F);
    }

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPhiEP", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPhiEPTH1F);
    }

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPtEM", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPtEMTH1F);
    }

    recHit = igetter.get(fmt::format("{}/{}RecHitMuonPhiEM", globalFolder_, i));
    if (recHit) {
      recHit->getTH1F()->Divide(NumberOfMuonPhiEMTH1F);
    }
  }
}
