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
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override{};  // Cannot use; it is called after dqmSaver

  void drawSummaryHistogram(edm::Service<DQMStore> &store);
  void createSummaryHist(edm::Service<DQMStore> &store,
                         MonitorElement *h2Src,
                         MonitorElement *&h2Sum,
                         std::vector<std::string> &listLayers);
  void createSummaryVFAT(edm::Service<DQMStore> &store,
                         MonitorElement *h2Src,
                         std::string strSuffix,
                         MonitorElement *&h2Sum);
  void refineSummaryHistogram(MonitorElement *h2Sum,
                              MonitorElement *h2SrcOcc,
                              MonitorElement *h2SrcCStatus,
                              MonitorElement *h2SrcMal = nullptr,
                              Bool_t bVarXBin = false);

  Float_t fReportSummary_;
  std::string strOutFile_;

  std::string strDirSummary_;
  std::string strDirStatus_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  fReportSummary_ = -1.0;
  strOutFile_ = cfg.getParameter<std::string>("fromFile");
  strDirSummary_ = "GEM/EventInfo";
  strDirStatus_ = "GEM/DAQStatus";
}

void GEMDQMHarvester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("fromFile", "");
  descriptions.add("GEMDQMHarvester", desc);
}

void GEMDQMHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                            DQMStore::IGetter &,
                                            edm::LuminosityBlock const &,
                                            edm::EventSetup const &) {
  edm::Service<DQMStore> store;
  drawSummaryHistogram(store);
}

void GEMDQMHarvester::drawSummaryHistogram(edm::Service<DQMStore> &store) {
  std::string strSrcDigiOcc = "GEM/digi/summaryOccDigi";
  std::string strSrcDigiMal = "GEM/digi/summaryMalfuncDigi";
  std::string strSrcCStatus = "GEM/DAQStatus/summaryStatus";

  std::string strSrcVFATOcc = "GEM/digi/digi_det";
  std::string strSrcVFATStatus = "GEM/DAQStatus/vfat_statusSum";

  store->setCurrentFolder(strDirSummary_);

  MonitorElement *h2SrcDigiOcc = store->get(strSrcDigiOcc);
  MonitorElement *h2SrcDigiMal = store->get(strSrcDigiMal);
  MonitorElement *h2SrcCStatus = store->get(strSrcCStatus);

  if (h2SrcDigiOcc != nullptr && h2SrcDigiMal != nullptr && h2SrcCStatus != nullptr) {
    MonitorElement *h2Sum = nullptr;
    std::vector<std::string> listLayer;
    createSummaryHist(store, h2SrcCStatus, h2Sum, listLayer);
    refineSummaryHistogram(h2Sum, h2SrcDigiOcc, h2SrcCStatus, h2SrcDigiMal, true);

    for (const auto &strSuffix : listLayer) {
      MonitorElement *h2SrcVFATOcc = store->get(strSrcVFATOcc + strSuffix);
      MonitorElement *h2SrcVFATStatus = store->get(strSrcVFATStatus + strSuffix);
      if (h2SrcVFATOcc == nullptr || h2SrcVFATStatus == nullptr)
        continue;
      MonitorElement *h2SumVFAT = nullptr;
      createSummaryVFAT(store, h2SrcVFATStatus, strSuffix, h2SumVFAT);
      refineSummaryHistogram(h2SumVFAT, h2SrcVFATOcc, h2SrcVFATStatus);
      h2SumVFAT->setTitle(h2SrcVFATStatus->getTitle());
      h2SumVFAT->setXTitle(h2SrcVFATStatus->getAxisTitle(1));
      h2SumVFAT->setYTitle(h2SrcVFATStatus->getAxisTitle(2));
    }
  }

  store->bookFloat("reportSummary")->Fill(fReportSummary_);
}

void GEMDQMHarvester::createSummaryHist(edm::Service<DQMStore> &store,
                                        MonitorElement *h2Src,
                                        MonitorElement *&h2Sum,
                                        std::vector<std::string> &listLayers) {
  store->setCurrentFolder(strDirSummary_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("reportSummaryMap", "", nBinX, 0.5, nBinX + 0.5, nBinY, 0.5, nBinY + 0.5);

  listLayers.clear();
  for (Int_t i = 1; i <= nBinX; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetXaxis()->GetBinLabel(i), 1);
  for (Int_t i = 1; i <= nBinY; i++) {
    std::string strLabelFull = h2Src->getTH2F()->GetYaxis()->GetBinLabel(i);
    auto nPos = strLabelFull.find(';');
    auto strLabel = strLabelFull.substr(0, nPos);
    listLayers.push_back(strLabelFull.substr(nPos + 1));
    h2Sum->setBinLabel(i, strLabel, 2);
  }
}

void GEMDQMHarvester::createSummaryVFAT(edm::Service<DQMStore> &store,
                                        MonitorElement *h2Src,
                                        std::string strSuffix,
                                        MonitorElement *&h2Sum) {
  store->setCurrentFolder(strDirStatus_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("vfat_statusSummary" + strSuffix, "", nBinX, 0.5, nBinX + 0.5, nBinY, 0.5, nBinY + 0.5);

  for (Int_t i = 1; i <= nBinX; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetXaxis()->GetBinLabel(i), 1);
  for (Int_t i = 1; i <= nBinY; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetYaxis()->GetBinLabel(i), 2);
}

// FIXME: Need more study about how to summarize
void GEMDQMHarvester::refineSummaryHistogram(MonitorElement *h2Sum,
                                             MonitorElement *h2SrcOcc,
                                             MonitorElement *h2SrcCStatus,
                                             MonitorElement *h2SrcMal,
                                             Bool_t bVarXBin) {
  Int_t nBinY = h2Sum->getNbinsY();
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = h2Sum->getNbinsX();
    if (bVarXBin) {
      nBinX = (Int_t)(h2SrcOcc->getBinContent(0, j) + 0.5);
      h2Sum->setBinContent(0, j, nBinX);
    }
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = h2SrcOcc->getBinContent(i, j);
      Float_t fStatus = h2SrcCStatus->getBinContent(i, j);
      Float_t fMal = (h2SrcMal != nullptr ? h2SrcMal->getBinContent(i, j) : 0.0);

      Float_t fRes = 0;
      if (fStatus > 0 || fMal > 0)
        fRes = 2;
      else if (fOcc > 0)
        fRes = 1;

      h2Sum->setBinContent(i, j, fRes);
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
