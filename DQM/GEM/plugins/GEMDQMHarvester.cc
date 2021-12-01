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
  Float_t refineSummaryHistogram(std::string strName,
                                 MonitorElement *h2Sum,
                                 MonitorElement *h2SrcOcc,
                                 MonitorElement *h2SrcAllNum,
                                 MonitorElement *h2SrcStatusE,
                                 MonitorElement *h2SrcStatusW);
  Int_t refineSummaryVFAT(std::string strName,
                          MonitorElement *h2Sum,
                          MonitorElement *h2SrcOcc,
                          MonitorElement *h2SrcStatusE,
                          MonitorElement *h2SrcStatusW);
  Int_t assessOneBin(
      std::string strName, Int_t nIdxX, Int_t nIdxY, Float_t fAll, Float_t fNumOcc, Float_t fNumWarn, Float_t fNumErr);

  Float_t fCutErr_, fCutLowErr_, fCutWarn_;

  const std::string strDirSummary_ = "GEM/EventInfo";
  const std::string strDirRecHit_ = "GEM/RecHits";
  const std::string strDirStatus_ = "GEM/DAQStatus";

  typedef std::vector<std::vector<Float_t>> TableStatusOcc;
  typedef std::vector<std::vector<Int_t>> TableStatusNum;

  std::vector<std::string> listLayer_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  fCutErr_ = cfg.getParameter<double>("cutErr");
  fCutLowErr_ = cfg.getParameter<double>("cutLowErr");
  fCutWarn_ = cfg.getParameter<double>("cutWarn");
}

void GEMDQMHarvester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("cutErr", 0.05);
  desc.add<double>("cutLowErr", 0.00);
  desc.add<double>("cutWarn", 0.05);
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
  Float_t fReportSummary = -1.0;

  std::string strSrcDigiOcc = "GEM/Digis/summaryOccDigi";
  std::string strSrcStatusA = "GEM/DAQStatus/chamberAllStatus";
  std::string strSrcStatusW = "GEM/DAQStatus/chamberWarnings";
  std::string strSrcStatusE = "GEM/DAQStatus/chamberErrors";

  std::string strSrcVFATOcc = "GEM/Digis/det";
  std::string strSrcVFATStatusW = "GEM/DAQStatus/vfat_statusWarnSum";
  std::string strSrcVFATStatusE = "GEM/DAQStatus/vfat_statusErrSum";

  store->setCurrentFolder(strDirSummary_);

  MonitorElement *h2SrcDigiOcc = store->get(strSrcDigiOcc);
  MonitorElement *h2SrcStatusA = store->get(strSrcStatusA);
  MonitorElement *h2SrcStatusW = store->get(strSrcStatusW);
  MonitorElement *h2SrcStatusE = store->get(strSrcStatusE);

  std::string strTitleSummary = "summary";

  if (h2SrcDigiOcc != nullptr && h2SrcStatusA != nullptr && h2SrcStatusW != nullptr && h2SrcStatusE != nullptr) {
    MonitorElement *h2Sum = nullptr;
    createSummaryHist(store, h2SrcStatusE, h2Sum, listLayer_);
    fReportSummary =
        refineSummaryHistogram(strTitleSummary, h2Sum, h2SrcDigiOcc, h2SrcStatusA, h2SrcStatusE, h2SrcStatusW);

    for (const auto &strSuffix : listLayer_) {
      MonitorElement *h2SrcVFATOcc = store->get(strSrcVFATOcc + strSuffix);
      MonitorElement *h2SrcVFATStatusW = store->get(strSrcVFATStatusW + strSuffix);
      MonitorElement *h2SrcVFATStatusE = store->get(strSrcVFATStatusE + strSuffix);
      if (h2SrcVFATOcc == nullptr || h2SrcVFATStatusW == nullptr || h2SrcVFATStatusE == nullptr)
        continue;

      MonitorElement *h2SumVFAT = nullptr;
      createSummaryVFAT(store, h2SrcVFATStatusE, strSuffix, h2SumVFAT);
      refineSummaryVFAT(strSuffix, h2SumVFAT, h2SrcVFATOcc, h2SrcVFATStatusE, h2SrcVFATStatusW);
      TString strNewTitle = h2SrcVFATStatusE->getTitle();
      h2SumVFAT->setTitle((const char *)strNewTitle.ReplaceAll("errors", "errors/warnings"));
      h2SumVFAT->setXTitle(h2SrcVFATStatusE->getAxisTitle(1));
      h2SumVFAT->setYTitle(h2SrcVFATStatusE->getAxisTitle(2));
    }
  }

  store->bookFloat("reportSummary")->Fill(fReportSummary);
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

Int_t GEMDQMHarvester::assessOneBin(
    std::string strName, Int_t nIdxX, Int_t nIdxY, Float_t fAll, Float_t fNumOcc, Float_t fNumWarn, Float_t fNumErr) {
  if (fNumErr > fCutErr_ * fAll)  // The error status criterion
    return 2;
  else if (fNumErr > fCutLowErr_ * fAll)  // The low-error status criterion
    return 4;
  else if (fNumWarn > fCutWarn_ * fAll)  // The warning status criterion
    return 3;
  else if (fNumOcc > 0)
    return 1;

  return 0;
}

// FIXME: Need more study about how to summarize
Float_t GEMDQMHarvester::refineSummaryHistogram(std::string strName,
                                                MonitorElement *h2Sum,
                                                MonitorElement *h2SrcOcc,
                                                MonitorElement *h2SrcStatusA,
                                                MonitorElement *h2SrcStatusE,
                                                MonitorElement *h2SrcStatusW) {
  Int_t nBinY = h2Sum->getNbinsY();
  Int_t nAllBin = 0, nFineBin = 0;
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = (Int_t)(h2SrcOcc->getBinContent(0, j) + 0.5);
    h2Sum->setBinContent(0, j, nBinX);
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = h2SrcOcc->getBinContent(i, j);
      Float_t fStatusAll = h2SrcStatusA->getBinContent(i, j);
      Float_t fStatusWarn = h2SrcStatusW->getBinContent(i, j);
      Float_t fStatusErr = h2SrcStatusE->getBinContent(i, j);

      Int_t nRes = assessOneBin(strName, i, j, fStatusAll, fOcc, fStatusWarn, fStatusErr);
      if (nRes == 1)
        nFineBin++;

      h2Sum->setBinContent(i, j, (Float_t)nRes);
      nAllBin++;
    }
  }

  return ((Float_t)nFineBin) / nAllBin;
}

Int_t GEMDQMHarvester::refineSummaryVFAT(std::string strName,
                                         MonitorElement *h2Sum,
                                         MonitorElement *h2SrcOcc,
                                         MonitorElement *h2SrcStatusE,
                                         MonitorElement *h2SrcStatusW) {
  Int_t nBinY = h2Sum->getNbinsY();
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = h2Sum->getNbinsX();
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = h2SrcOcc->getBinContent(i, j);
      Float_t fStatusWarn = h2SrcStatusW->getBinContent(i, j);
      Float_t fStatusErr = h2SrcStatusE->getBinContent(i, j);
      Float_t fStatusAll = fOcc + fStatusWarn + fStatusErr;

      Int_t nRes = assessOneBin(strName, i, j, fStatusAll, fOcc, fStatusWarn, fStatusErr);
      h2Sum->setBinContent(i, j, (Float_t)nRes);
    }
  }

  return 0;
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
