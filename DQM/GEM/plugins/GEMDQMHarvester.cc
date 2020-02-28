#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <TH2F.h>
#include <TFile.h>
#include <TDirectoryFile.h>
#include <TKey.h>

using namespace edm;

class GEMDQMHarvester : public DQMEDHarvester {
public:
  GEMDQMHarvester(const edm::ParameterSet &);
  ~GEMDQMHarvester() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void endRun(edm::Run const &, edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override{};  // Cannot use; it is called after dqmSaver

  void refineSummaryHistogram(edm::Service<DQMStore> &);
  void refineSummaryHistogramCore(TH3F *, std::string &, TH2F *&, std::string strTmpPrefix = "tmp_");

  void fillUnderOverflowBunchCrossing(edm::Service<DQMStore> &, std::string);

  std::string strOutFile_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  strOutFile_ = cfg.getParameter<std::string>("fromFile");
}

void GEMDQMHarvester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("fromFile", "");
  descriptions.add("GEMDQMHarvester", desc);
}

void GEMDQMHarvester::endRun(edm::Run const &, edm::EventSetup const &) {
  edm::Service<DQMStore> store;
  refineSummaryHistogram(store);

  store->setCurrentFolder("GEM/StatusDigi");
  auto listME = store->getMEs();

  for (auto strName : listME) {
    if (strName.find("vfatStatus_BC_") != std::string::npos) {
      fillUnderOverflowBunchCrossing(store, strName);
    }
  }
}

void GEMDQMHarvester::refineSummaryHistogram(edm::Service<DQMStore> &store) {
  std::string strDirCurr = "GEM/EventInfo";
  std::string strNameSrc = "reportSummaryMapPreliminary";
  std::string strNewName = "reportSummaryMap";

  store->setCurrentFolder(strDirCurr);

  MonitorElement *h3Curr = store->get(strDirCurr + "/" + strNameSrc);
  TH2F *h2New = nullptr;

  refineSummaryHistogramCore(h3Curr->getTH3F(), strNewName, h2New);
  store->book2D(strNewName, h2New);
}

void GEMDQMHarvester::refineSummaryHistogramCore(TH3F *h3Src,
                                                 std::string &strNewName,
                                                 TH2F *&h2New,
                                                 std::string strTmpPrefix) {
  Int_t i, j;

  Int_t nNBinX = h3Src->GetNbinsX();
  Int_t nNBinY = h3Src->GetNbinsY();

  Float_t arrfBinX[128], arrfBinY[32];

  for (i = 0; i <= nNBinX; i++)
    arrfBinX[i] = h3Src->GetXaxis()->GetBinLowEdge(i + 1);
  for (i = 0; i <= nNBinY; i++)
    arrfBinY[i] = h3Src->GetYaxis()->GetBinLowEdge(i + 1);

  h2New = new TH2F(strNewName.c_str(), h3Src->GetTitle(), nNBinX, arrfBinX, nNBinY, arrfBinY);

  for (i = 0; i < nNBinX; i++) {
    h2New->GetXaxis()->SetBinLabel(i + 1, h3Src->GetXaxis()->GetBinLabel(i + 1));
    for (j = 0; j < nNBinY; j++) {
      h2New->GetYaxis()->SetBinLabel(j + 1, h3Src->GetYaxis()->GetBinLabel(j + 1));

      if (h3Src->GetBinContent(i + 1, j + 1, 2) != 0) {
        h2New->SetBinContent(i + 1, j + 1, 2);
      } else if (h3Src->GetBinContent(i + 1, j + 1, 1) != 0) {
        h2New->SetBinContent(i + 1, j + 1, 1);
      }
    }
  }
}

void GEMDQMHarvester::fillUnderOverflowBunchCrossing(edm::Service<DQMStore> &store, std::string strNameSrc) {
  std::string strDirCurr = "GEM/StatusDigi";

  store->setCurrentFolder(strDirCurr);
  MonitorElement *h2Curr = store->get(strDirCurr + "/" + strNameSrc);

  Int_t nNBinX = h2Curr->getNbinsX();
  Int_t nNBinY = h2Curr->getNbinsY();

  for (Int_t i = 0; i < nNBinY; i++) {
    h2Curr->setBinContent(1, i, h2Curr->getBinContent(0, i) + h2Curr->getBinContent(1, i));
    h2Curr->setBinContent(nNBinX, i, h2Curr->getBinContent(nNBinX, i) + h2Curr->getBinContent(nNBinX + 1, i));
  }
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
