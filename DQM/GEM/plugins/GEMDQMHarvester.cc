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
  void createSummaryHist(edm::Service<DQMStore> &store, MonitorElement *h2Src, MonitorElement *&h2Sum);
  void refineSummaryHistogram(MonitorElement *h2Sum,
                              MonitorElement *h2SrcDigiOcc,
                              MonitorElement *h2SrcDigiMal,
                              MonitorElement *h2SrcCStatus);

  Float_t fReportSummary_;
  std::string strOutFile_;

  std::string strDirSummary_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  fReportSummary_ = -1.0;
  strOutFile_ = cfg.getParameter<std::string>("fromFile");
  strDirSummary_ = "GEM/EventInfo";
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

void GEMDQMHarvester::createSummaryHist(edm::Service<DQMStore> &store, MonitorElement *h2Src, MonitorElement *&h2Sum) {
  store->setCurrentFolder(strDirSummary_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("reportSummaryMap", "", nBinX, 0.5, nBinX + 0.5, nBinY, 0.5, nBinY + 0.5);

  for (Int_t i = 1; i <= nBinX; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetXaxis()->GetBinLabel(i), 1);
  for (Int_t i = 1; i <= nBinY; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetYaxis()->GetBinLabel(i), 2);
}

void GEMDQMHarvester::drawSummaryHistogram(edm::Service<DQMStore> &store) {
  std::string strSrcDigiOcc = "GEM/digi/summaryOccDigi";
  std::string strSrcDigiMal = "GEM/digi/summaryMalfuncDigi";
  std::string strSrcCStatus = "GEM/DAQStatus/summaryStatus";

  store->setCurrentFolder(strDirSummary_);

  MonitorElement *h2SrcDigiOcc = store->get(strSrcDigiOcc);
  MonitorElement *h2SrcDigiMal = store->get(strSrcDigiMal);
  MonitorElement *h2SrcCStatus = store->get(strSrcCStatus);

  if (h2SrcDigiOcc != nullptr && h2SrcDigiMal != nullptr && h2SrcCStatus != nullptr) {
    MonitorElement *h2Sum = nullptr;
    createSummaryHist(store, h2SrcCStatus, h2Sum);
    refineSummaryHistogram(h2Sum, h2SrcDigiOcc, h2SrcDigiMal, h2SrcCStatus);
  }

  store->bookFloat("reportSummary")->Fill(fReportSummary_);
}

// FIXME: Need more study about how to summarize
void GEMDQMHarvester::refineSummaryHistogram(MonitorElement *h2Sum,
                                             MonitorElement *h2SrcDigiOcc,
                                             MonitorElement *h2SrcDigiMal,
                                             MonitorElement *h2SrcCStatus) {
  Int_t nBinY = h2Sum->getNbinsY();
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = (Int_t)(h2SrcDigiOcc->getBinContent(0, j) + 0.5);
    h2Sum->setBinContent(0, j, nBinX);
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fDigiOcc = h2SrcDigiOcc->getBinContent(i, j);
      Float_t fDigiMal = h2SrcDigiMal->getBinContent(i, j);
      Float_t fCStatus = h2SrcCStatus->getBinContent(i, j);

      Float_t fRes = 0;
      if (fCStatus > 0 || fDigiMal > 0)
        fRes = 2;
      else if (fDigiOcc > 0)
        fRes = 1;

      h2Sum->setBinContent(i, j, fRes);
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
