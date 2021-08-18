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
  void copyLabels(MonitorElement *h2Src, MonitorElement *h2Dst);
  void createSummaryHist(edm::Service<DQMStore> &store,
                         MonitorElement *h2Src,
                         MonitorElement *&h2Sum,
                         std::vector<std::string> &listLayers);
  void createSummaryVFAT(edm::Service<DQMStore> &store,
                         MonitorElement *h2Src,
                         std::string strSuffix,
                         MonitorElement *&h2Sum);
  Float_t refineSummaryHistogram(MonitorElement *h2Sum,
                                 MonitorElement *h2SrcOcc,
                                 MonitorElement *h2SrcStatusE,
                                 MonitorElement *h2SrcStatusW = nullptr,
                                 Bool_t bVarXBin = false);

  Float_t fReportSummary_;
  std::string strOutFile_;

  std::string strDirSummary_;
  std::string strDirRecHit_;
  std::string strDirStatus_;

  std::vector<std::string> listLayer_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  fReportSummary_ = -1.0;
  strOutFile_ = cfg.getParameter<std::string>("fromFile");
  strDirSummary_ = "GEM/EventInfo";
  strDirRecHit_ = "GEM/RecHits";
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
  std::string strSrcDigiOcc = "GEM/Digis/summaryOccDigi";
  std::string strSrcStatusW = "GEM/DAQStatus/chamberWarnings";
  std::string strSrcStatusE = "GEM/DAQStatus/chamberErrors";

  std::string strSrcVFATOcc = "GEM/Digis/det";
  std::string strSrcVFATStatusW = "GEM/DAQStatus/vfat_statusWarnSum";
  std::string strSrcVFATStatusE = "GEM/DAQStatus/vfat_statusErrSum";

  store->setCurrentFolder(strDirSummary_);

  MonitorElement *h2SrcDigiOcc = store->get(strSrcDigiOcc);
  MonitorElement *h2SrcStatusW = store->get(strSrcStatusW);
  MonitorElement *h2SrcStatusE = store->get(strSrcStatusE);

  if (h2SrcDigiOcc != nullptr && h2SrcStatusW != nullptr && h2SrcStatusE != nullptr) {
    MonitorElement *h2Sum = nullptr;
    createSummaryHist(store, h2SrcStatusE, h2Sum, listLayer_);
    fReportSummary_ = refineSummaryHistogram(h2Sum, h2SrcDigiOcc, h2SrcStatusE, h2SrcStatusW, true);

    for (const auto &strSuffix : listLayer_) {
      MonitorElement *h2SrcVFATOcc = store->get(strSrcVFATOcc + strSuffix);
      MonitorElement *h2SrcVFATStatusW = store->get(strSrcVFATStatusW + strSuffix);
      MonitorElement *h2SrcVFATStatusE = store->get(strSrcVFATStatusE + strSuffix);
      if (h2SrcVFATOcc == nullptr || h2SrcVFATStatusW == nullptr || h2SrcVFATStatusE == nullptr)
        continue;
      MonitorElement *h2SumVFAT = nullptr;
      createSummaryVFAT(store, h2SrcVFATStatusE, strSuffix, h2SumVFAT);
      refineSummaryHistogram(h2SumVFAT, h2SrcVFATOcc, h2SrcVFATStatusE, h2SrcVFATStatusW);
      TString strNewTitle = h2SrcVFATStatusE->getTitle();
      h2SumVFAT->setTitle((const char *)strNewTitle.ReplaceAll("errors", "errors/warnings"));
      h2SumVFAT->setXTitle(h2SrcVFATStatusE->getAxisTitle(1));
      h2SumVFAT->setYTitle(h2SrcVFATStatusE->getAxisTitle(2));
    }
  }

  store->bookFloat("reportSummary")->Fill(fReportSummary_);
}

void GEMDQMHarvester::copyLabels(MonitorElement *h2Src, MonitorElement *h2Dst) {
  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();

  for (Int_t i = 1; i <= nBinX; i++) {
    h2Dst->setBinLabel(i, h2Src->getTH2F()->GetXaxis()->GetBinLabel(i), 1);
  }
  for (Int_t i = 1; i <= nBinY; i++) {
    h2Dst->setBinLabel(i, h2Src->getTH2F()->GetYaxis()->GetBinLabel(i), 2);
  }
  h2Dst->setTitle(h2Src->getTitle());
  h2Dst->setXTitle(h2Src->getAxisTitle(1));
  h2Dst->setYTitle(h2Src->getAxisTitle(2));
}

void GEMDQMHarvester::createSummaryHist(edm::Service<DQMStore> &store,
                                        MonitorElement *h2Src,
                                        MonitorElement *&h2Sum,
                                        std::vector<std::string> &listLayers) {
  //store->setCurrentFolder(strDirSummary_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("reportSummaryMap", "", nBinX, 0.5, nBinX + 0.5, nBinY, 0.5, nBinY + 0.5);
  h2Sum->setTitle("Summary plot");
  h2Sum->setXTitle("Chamber");
  h2Sum->setYTitle("Layer");

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
  //store->setCurrentFolder(strDirStatus_);
  //store->setCurrentFolder(strDirSummary_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("vfat_statusSummary" + strSuffix, "", nBinX, 0.5, nBinX + 0.5, nBinY, -0.5, nBinY - 0.5);
  copyLabels(h2Src, h2Sum);
}

// FIXME: Need more study about how to summarize
Float_t GEMDQMHarvester::refineSummaryHistogram(MonitorElement *h2Sum,
                                                MonitorElement *h2SrcOcc,
                                                MonitorElement *h2SrcStatusE,
                                                MonitorElement *h2SrcStatusW,
                                                Bool_t bVarXBin) {
  Int_t nBinY = h2Sum->getNbinsY();
  Int_t nAllBin = 0, nFineBin = 0;
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = h2Sum->getNbinsX();
    if (bVarXBin) {
      nBinX = (Int_t)(h2SrcOcc->getBinContent(0, j) + 0.5);
      h2Sum->setBinContent(0, j, nBinX);
    }
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = h2SrcOcc->getBinContent(i, j);
      Float_t fStatusWarn = (h2SrcStatusW != nullptr ? h2SrcStatusW->getBinContent(i, j) : 0.0);
      Float_t fStatusErr = h2SrcStatusE->getBinContent(i, j);

      Float_t fRes = 0;
      if (fStatusErr > 0)
        fRes = 2;
      else if (fStatusWarn > 0)
        fRes = 3;
      else if (fOcc > 0) {
        fRes = 1;
        nFineBin++;
      }

      h2Sum->setBinContent(i, j, fRes);
      nAllBin++;
    }
  }

  return ((Float_t)nFineBin) / nAllBin;
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
