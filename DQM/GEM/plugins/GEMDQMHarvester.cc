#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  typedef std::tuple<int, int> IdChamber;
  typedef std::tuple<int, int, int> IdVFAT;
  typedef struct PreStatusInfo {
    int nLumiStart;
    int nLumiEnd;
    int nStatus;
  } StatusInfo;

  class NumStatus {
  public:
    NumStatus(Float_t fNumTotal = 0,
              Float_t fNumOcc = 0,
              Float_t fNumErrVFAT = 0,
              Float_t fNumWarnVFAT = 0,
              Float_t fNumErrOH = 0,
              Float_t fNumWarnOH = 0,
              Float_t fNumErrAMC = 0,
              Float_t fNumWarnAMC = 0,
              Float_t fNumErrAMC13 = 0)
        : fNumTotal_(fNumTotal),
          fNumOcc_(fNumOcc),
          fNumErrVFAT_(fNumErrVFAT),
          fNumWarnVFAT_(fNumWarnVFAT),
          fNumErrOH_(fNumErrOH),
          fNumWarnOH_(fNumWarnOH),
          fNumErrAMC_(fNumErrAMC),
          fNumWarnAMC_(fNumWarnAMC),
          fNumErrAMC13_(fNumErrAMC13) {}
    float fNumTotal_;
    float fNumOcc_;
    float fNumErrVFAT_;
    float fNumWarnVFAT_;
    float fNumErrOH_;
    float fNumWarnOH_;
    float fNumErrAMC_;
    float fNumWarnAMC_;
    float fNumErrAMC13_;
  };

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &iLumi,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override{};  // Cannot use; it is called after dqmSaver

  void drawSummaryHistogram(edm::Service<DQMStore> &store, Int_t nLumiCurr);
  void createTableWatchingSummary();
  void copyLabels(MonitorElement *h2Src, MonitorElement *h2Dst);
  void getGeometryInfo(edm::Service<DQMStore> &store, MonitorElement *h2Src);
  void createSummaryHist(edm::Service<DQMStore> &store, MonitorElement *h2Src, MonitorElement *&h2Sum);
  void createSummaryVFAT(edm::Service<DQMStore> &store,
                         MonitorElement *h2Src,
                         std::string strSuffix,
                         MonitorElement *&h2Sum);
  Float_t refineSummaryHistogram(std::string strName,
                                 MonitorElement *h2Sum,
                                 std::vector<MonitorElement *> &listOccPlots,
                                 MonitorElement *h2SrcStatusA,
                                 MonitorElement *h2SrcStatusE,
                                 MonitorElement *h2SrcStatusW,
                                 MonitorElement *h2SrcStatusEVFAT,
                                 MonitorElement *h2SrcStatusWVFAT,
                                 MonitorElement *h2SrcStatusEOH,
                                 MonitorElement *h2SrcStatusWOH,
                                 MonitorElement *h2SrcStatusEAMC,
                                 MonitorElement *h2SrcStatusWAMC,
                                 MonitorElement *h2SrcStatusEAMC13,
                                 Int_t nLumiCurr);
  Int_t refineSummaryVFAT(std::string strName,
                          MonitorElement *h2Sum,
                          MonitorElement *h2SrcOcc,
                          MonitorElement *h2SrcStatusE,
                          MonitorElement *h2SrcStatusW,
                          Int_t nLumiCurr,
                          Int_t nIdxLayer);
  Int_t assessOneBin(
      std::string strName, Int_t nIdxX, Int_t nIdxY, Float_t fAll, Float_t fNumOcc, Float_t fNumErr, Float_t fNumWarn);

  Int_t UpdateStatusChamber(Int_t nIdxLayer, Int_t nIdxCh, Int_t nLumiCurr, NumStatus numStatus);
  Int_t UpdateStatusChamber(Int_t nIdxLayer, Int_t nIdxCh, Int_t nIdxVFAT, Int_t nLumiCurr, NumStatus numStatus);
  Int_t UpdateStatusChamber(std::vector<StatusInfo> &listStatus,
                            NumStatus &numStatus,
                            Int_t nLumiCurr,
                            NumStatus numStatusNew);
  void createLumiFuncHist(edm::Service<DQMStore> &store, std::string strSuffix, Int_t nIdxLayer, Int_t nLumiCurr);
  void createInactiveChannelFracHist(edm::Service<DQMStore> &store, std::string strSuffix, Int_t nNumChamber);

  Float_t fCutErr_, fCutLowErr_, fCutWarn_;

  const std::string strDirSummary_ = "GEM/EventInfo";
  const std::string strDirRecHit_ = "GEM/RecHits";
  const std::string strDirStatus_ = "GEM/DAQStatus";

  const Int_t nCodeFine_ = 1;
  const Int_t nCodeError_ = 2;
  const Int_t nCodeWarning_ = 3;
  const Int_t nCodeLowError_ = 4;

  const Int_t nBitWarnVFAT_ = 7;
  const Int_t nBitErrVFAT_ = 6;
  const Int_t nBitWarnOH_ = 5;
  const Int_t nBitErrOH_ = 4;
  const Int_t nBitWarnAMC_ = 3;
  const Int_t nBitErrAMC_ = 2;
  const Int_t nBitErrAMC13_ = 1;
  const Int_t nBitOcc_ = 0;

  const Int_t nNumVFATs_ = 24;

  const Int_t nMaxLumi_ = 6000;  // From DQMServices/Components/plugins/DQMProvInfo.h
  //const Int_t nResolutionLumi_ = 5;
  Int_t nResolutionLumi_;

  typedef std::vector<std::vector<Float_t>> TableStatusOcc;
  typedef std::vector<std::vector<Int_t>> TableStatusNum;

  std::map<IdChamber, std::vector<StatusInfo>> mapStatusChambersSummary_;
  std::map<IdVFAT, std::vector<StatusInfo>> mapStatusVFATsSummary_;
  std::map<IdChamber, NumStatus> mapNumStatusChambersSummary_;
  std::map<IdVFAT, NumStatus> mapNumStatusVFATsSummary_;

  std::vector<std::string> listLayer_;
  std::map<std::string, int> mapIdxLayer_;  // All indices in the following objects start at 1
  std::map<int, int> mapNumChPerChamber_;
  std::map<int, MonitorElement *> mapHistLumiFunc_;
  Bool_t bIsStatusChambersInit_;
};

GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  fCutErr_ = cfg.getParameter<double>("cutErr");
  fCutLowErr_ = cfg.getParameter<double>("cutLowErr");
  fCutWarn_ = cfg.getParameter<double>("cutWarn");
  nResolutionLumi_ = cfg.getParameter<int>("resolutionLumi");
  bIsStatusChambersInit_ = false;
}

void GEMDQMHarvester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("cutErr", 0.05);
  desc.add<double>("cutLowErr", 0.00);
  desc.add<double>("cutWarn", 0.05);
  desc.add<int>("resolutionLumi", 1);
  descriptions.add("GEMDQMHarvester", desc);
}

void GEMDQMHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                            DQMStore::IGetter &,
                                            edm::LuminosityBlock const &iLumi,
                                            edm::EventSetup const &) {
  edm::Service<DQMStore> store;
  Int_t nLumiCurr = iLumi.id().luminosityBlock();
  drawSummaryHistogram(store, nLumiCurr);
}

void GEMDQMHarvester::drawSummaryHistogram(edm::Service<DQMStore> &store, Int_t nLumiCurr) {
  Float_t fReportSummary = -1.0;

  std::string strSrcStatusA = "GEM/DAQStatus/chamberAllStatus";
  std::string strSrcStatusE = "GEM/DAQStatus/chamberErrors";
  std::string strSrcStatusW = "GEM/DAQStatus/chamberWarnings";
  std::string strSrcStatusEVFAT = "GEM/DAQStatus/chamberVFATErrors";
  std::string strSrcStatusWVFAT = "GEM/DAQStatus/chamberVFATWarnings";
  std::string strSrcStatusEOH = "GEM/DAQStatus/chamberOHErrors";
  std::string strSrcStatusWOH = "GEM/DAQStatus/chamberOHWarnings";
  std::string strSrcStatusEAMC = "GEM/DAQStatus/chamberAMCErrors";
  std::string strSrcStatusWAMC = "GEM/DAQStatus/chamberAMCWarnings";
  std::string strSrcStatusEAMC13 = "GEM/DAQStatus/chamberAMC13Errors";

  std::string strSrcVFATOcc = "GEM/Digis/occ";
  std::string strSrcVFATStatusW = "GEM/DAQStatus/vfat_statusWarnSum";
  std::string strSrcVFATStatusE = "GEM/DAQStatus/vfat_statusErrSum";

  store->setCurrentFolder(strDirSummary_);

  MonitorElement *h2SrcStatusA = store->get(strSrcStatusA);
  MonitorElement *h2SrcStatusE = store->get(strSrcStatusE);
  MonitorElement *h2SrcStatusW = store->get(strSrcStatusW);
  MonitorElement *h2SrcStatusEVFAT = store->get(strSrcStatusEVFAT);
  MonitorElement *h2SrcStatusWVFAT = store->get(strSrcStatusWVFAT);
  MonitorElement *h2SrcStatusEOH = store->get(strSrcStatusEOH);
  MonitorElement *h2SrcStatusWOH = store->get(strSrcStatusWOH);
  MonitorElement *h2SrcStatusEAMC = store->get(strSrcStatusEAMC);
  MonitorElement *h2SrcStatusWAMC = store->get(strSrcStatusWAMC);
  MonitorElement *h2SrcStatusEAMC13 = store->get(strSrcStatusEAMC13);

  std::string strTitleSummary = "summary";

  getGeometryInfo(store, h2SrcStatusEOH);

  if (h2SrcStatusA != nullptr && h2SrcStatusE != nullptr && h2SrcStatusW != nullptr && h2SrcStatusEVFAT != nullptr &&
      h2SrcStatusWVFAT != nullptr && h2SrcStatusEOH != nullptr && h2SrcStatusWOH != nullptr &&
      h2SrcStatusEAMC != nullptr && h2SrcStatusWAMC != nullptr && h2SrcStatusEAMC13 != nullptr) {
    MonitorElement *h2Sum = nullptr;
    createSummaryHist(store, h2SrcStatusEOH, h2Sum);
    createTableWatchingSummary();

    std::vector<MonitorElement *> listOccPlots(listLayer_.size() + 1);  // The index starts at 1
    for (const auto &strSuffix : listLayer_) {
      if (mapIdxLayer_.find(strSuffix) == mapIdxLayer_.end())
        continue;
      auto nIdxLayer = mapIdxLayer_[strSuffix];
      MonitorElement *h2SrcVFATOcc = store->get(strSrcVFATOcc + strSuffix);
      if (h2SrcVFATOcc == nullptr)
        continue;
      listOccPlots[nIdxLayer] = h2SrcVFATOcc;
    }

    fReportSummary = refineSummaryHistogram(strTitleSummary,
                                            h2Sum,
                                            listOccPlots,
                                            h2SrcStatusA,
                                            h2SrcStatusE,
                                            h2SrcStatusW,
                                            h2SrcStatusEVFAT,
                                            h2SrcStatusWVFAT,
                                            h2SrcStatusEOH,
                                            h2SrcStatusWOH,
                                            h2SrcStatusEAMC,
                                            h2SrcStatusWAMC,
                                            h2SrcStatusEAMC13,
                                            nLumiCurr);

    for (const auto &strSuffix : listLayer_) {
      if (mapIdxLayer_.find(strSuffix) == mapIdxLayer_.end())
        continue;
      auto nIdxLayer = mapIdxLayer_[strSuffix];
      MonitorElement *h2SrcVFATOcc = store->get(strSrcVFATOcc + strSuffix);
      MonitorElement *h2SrcVFATStatusW = store->get(strSrcVFATStatusW + strSuffix);
      MonitorElement *h2SrcVFATStatusE = store->get(strSrcVFATStatusE + strSuffix);
      if (h2SrcVFATOcc == nullptr || h2SrcVFATStatusW == nullptr || h2SrcVFATStatusE == nullptr)
        continue;

      MonitorElement *h2SumVFAT = nullptr;
      createSummaryVFAT(store, h2SrcVFATStatusE, strSuffix, h2SumVFAT);
      refineSummaryVFAT(strSuffix, h2SumVFAT, h2SrcVFATOcc, h2SrcVFATStatusE, h2SrcVFATStatusW, nLumiCurr, nIdxLayer);
      TString strNewTitle = h2SrcVFATStatusE->getTitle();
      h2SumVFAT->setTitle((const char *)strNewTitle.ReplaceAll("errors", "errors/warnings"));
      h2SumVFAT->setXTitle(h2SrcVFATStatusE->getAxisTitle(1));
      h2SumVFAT->setYTitle(h2SrcVFATStatusE->getAxisTitle(2));

      createLumiFuncHist(store, strSuffix, nIdxLayer, nLumiCurr);
    }
  }

  for (const auto &strSuffix : listLayer_) {
    if (mapIdxLayer_.find(strSuffix) == mapIdxLayer_.end())
      continue;
    auto nNumChamber = mapNumChPerChamber_[mapIdxLayer_[strSuffix]];
    createInactiveChannelFracHist(store, strSuffix, nNumChamber);
  }

  store->bookFloat("reportSummary")->Fill(fReportSummary);
}

void GEMDQMHarvester::createTableWatchingSummary() {
  if (bIsStatusChambersInit_)
    return;

  for (const auto &[nIdxLayer, nNumCh] : mapNumChPerChamber_) {
    for (Int_t i = 1; i <= nNumCh; i++) {
      mapStatusChambersSummary_[{nIdxLayer, i}] = std::vector<StatusInfo>();
      mapNumStatusChambersSummary_[{nIdxLayer, i}] = NumStatus();
      for (Int_t j = 1; j <= nNumVFATs_; j++) {
        mapStatusVFATsSummary_[{nIdxLayer, i, j}] = std::vector<StatusInfo>();
        mapNumStatusVFATsSummary_[{nIdxLayer, i, j}] = NumStatus();
      }
    }
  }

  bIsStatusChambersInit_ = true;
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

void GEMDQMHarvester::getGeometryInfo(edm::Service<DQMStore> &store, MonitorElement *h2Src) {
  listLayer_.clear();
  mapIdxLayer_.clear();
  mapNumChPerChamber_.clear();

  if (h2Src != nullptr) {  // For online and offline
    Int_t nBinY = h2Src->getNbinsY();
    listLayer_.push_back("");

    for (Int_t i = 1; i <= nBinY; i++) {
      std::string strLabelFull = h2Src->getTH2F()->GetYaxis()->GetBinLabel(i);
      Int_t nBinXActual = (Int_t)(h2Src->getBinContent(0, i) + 0.5);
      auto nPos = strLabelFull.find(';');
      auto strLayer = strLabelFull.substr(nPos + 1);
      listLayer_.push_back(strLayer);
      mapIdxLayer_[strLayer] = i;
      mapNumChPerChamber_[i] = nBinXActual;
    }
  } else {  // For others (validation and...?)
    listLayer_.push_back("");
    if (store->get("GEM/Digis/occupancy_GE11-M-L1/occ_GE11-M-01L1-S") != nullptr) {
      listLayer_.push_back("_GE11-P-L2");
      listLayer_.push_back("_GE11-P-L1");
      listLayer_.push_back("_GE11-M-L1");
      listLayer_.push_back("_GE11-M-L2");
      mapIdxLayer_["_GE11-P-L2"] = 1;
      mapIdxLayer_["_GE11-P-L1"] = 2;
      mapIdxLayer_["_GE11-M-L1"] = 3;
      mapIdxLayer_["_GE11-M-L2"] = 4;
      mapNumChPerChamber_[1] = 36;
      mapNumChPerChamber_[2] = 36;
      mapNumChPerChamber_[3] = 36;
      mapNumChPerChamber_[4] = 36;
    }
    // FIXME: How about GE21 and ME0?
  }
}

void GEMDQMHarvester::createSummaryHist(edm::Service<DQMStore> &store, MonitorElement *h2Src, MonitorElement *&h2Sum) {
  //store->setCurrentFolder(strDirSummary_);

  Int_t nBinX = h2Src->getNbinsX(), nBinY = h2Src->getNbinsY();
  h2Sum = store->book2D("reportSummaryMap", "", nBinX, 0.5, nBinX + 0.5, nBinY, 0.5, nBinY + 0.5);
  h2Sum->setTitle("Summary plot");
  h2Sum->setXTitle("Chamber");
  h2Sum->setYTitle("Layer");

  for (Int_t i = 1; i <= nBinX; i++)
    h2Sum->setBinLabel(i, h2Src->getTH2F()->GetXaxis()->GetBinLabel(i), 1);
  for (Int_t i = 1; i <= nBinY; i++)
    h2Sum->setBinLabel(i, listLayer_[i].substr(1), 2);
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
    std::string strName, Int_t nIdxX, Int_t nIdxY, Float_t fAll, Float_t fNumOcc, Float_t fNumErr, Float_t fNumWarn) {
  if (fNumErr > fCutErr_ * fAll)  // The error status criterion
    return nCodeError_;
  else if (fNumErr > fCutLowErr_ * fAll)  // The low-error status criterion
    return nCodeLowError_;
  else if (fNumWarn > fCutWarn_ * fAll)  // The warning status criterion
    return nCodeWarning_;
  else if (fNumOcc > 0)
    return nCodeFine_;

  return 0;
}

// FIXME: Need more study about how to summarize
Float_t GEMDQMHarvester::refineSummaryHistogram(std::string strName,
                                                MonitorElement *h2Sum,
                                                std::vector<MonitorElement *> &listOccPlots,
                                                MonitorElement *h2SrcStatusA,
                                                MonitorElement *h2SrcStatusE,
                                                MonitorElement *h2SrcStatusW,
                                                MonitorElement *h2SrcStatusEVFAT,
                                                MonitorElement *h2SrcStatusWVFAT,
                                                MonitorElement *h2SrcStatusEOH,
                                                MonitorElement *h2SrcStatusWOH,
                                                MonitorElement *h2SrcStatusEAMC,
                                                MonitorElement *h2SrcStatusWAMC,
                                                MonitorElement *h2SrcStatusEAMC13,
                                                Int_t nLumiCurr) {
  Int_t nBinY = h2Sum->getNbinsY();
  Int_t nAllBin = 0, nFineBin = 0;
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = (Int_t)(h2SrcStatusE->getBinContent(0, j) + 0.5);
    auto h2SrcOcc = listOccPlots[j];
    Int_t nBinYOcc = 0;
    if (h2SrcOcc != nullptr) {
      nBinYOcc = h2SrcOcc->getNbinsY();
    }

    h2Sum->setBinContent(0, j, nBinX);
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = 0;
      for (Int_t r = 1; r <= nBinYOcc; r++) {
        fOcc += h2SrcOcc->getBinContent(i, r);
      }

      Float_t fStatusAll = h2SrcStatusA->getBinContent(i, j);
      Float_t fStatusErr = h2SrcStatusE->getBinContent(i, j);
      Float_t fStatusWarn = h2SrcStatusW->getBinContent(i, j);
      Float_t fStatusErrVFAT = h2SrcStatusEVFAT->getBinContent(i, j);
      Float_t fStatusWarnVFAT = h2SrcStatusWVFAT->getBinContent(i, j);
      Float_t fStatusErrOH = h2SrcStatusEOH->getBinContent(i, j);
      Float_t fStatusWarnOH = h2SrcStatusWOH->getBinContent(i, j);
      Float_t fStatusErrAMC = h2SrcStatusEAMC->getBinContent(i, j);
      Float_t fStatusWarnAMC = h2SrcStatusWAMC->getBinContent(i, j);
      Float_t fStatusErrAMC13 = h2SrcStatusEAMC13->getBinContent(i, j);
      NumStatus numStatus(fStatusAll,
                          fOcc,
                          fStatusErrVFAT,
                          fStatusWarnVFAT,
                          fStatusErrOH,
                          fStatusWarnOH,
                          fStatusErrAMC,
                          fStatusWarnAMC,
                          fStatusErrAMC13);
      UpdateStatusChamber(j, i, nLumiCurr, numStatus);

      Int_t nRes = assessOneBin(strName, i, j, fStatusAll, fOcc, fStatusErr, fStatusWarn);
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
                                         MonitorElement *h2SrcStatusW,
                                         Int_t nLumiCurr,
                                         Int_t nIdxLayer) {
  Int_t nBinY = h2Sum->getNbinsY();
  for (Int_t j = 1; j <= nBinY; j++) {
    Int_t nBinX = h2Sum->getNbinsX();
    for (Int_t i = 1; i <= nBinX; i++) {
      Float_t fOcc = h2SrcOcc->getBinContent(i, j);
      Float_t fStatusErr = h2SrcStatusE->getBinContent(i, j);
      Float_t fStatusWarn = h2SrcStatusW->getBinContent(i, j);
      Float_t fStatusAll = fOcc + fStatusErr + fStatusWarn;
      NumStatus numStatus(fStatusAll, fOcc, fStatusErr, fStatusWarn, 0, 0, 0, 0, 0);
      UpdateStatusChamber(nIdxLayer, i, j, nLumiCurr, numStatus);

      Int_t nRes = assessOneBin(strName, i, j, fStatusAll, fOcc, fStatusErr, fStatusWarn);
      h2Sum->setBinContent(i, j, (Float_t)nRes);
    }
  }

  return 0;
}

Int_t GEMDQMHarvester::UpdateStatusChamber(Int_t nIdxLayer, Int_t nIdxCh, Int_t nLumiCurr, NumStatus numStatus) {
  if (!bIsStatusChambersInit_)
    return 0;
  if (0 >= nIdxCh || nIdxCh > mapNumChPerChamber_[nIdxLayer])
    return 0;
  auto &listStatus = mapStatusChambersSummary_[{nIdxLayer, nIdxCh}];
  auto &numStatusPrev = mapNumStatusChambersSummary_[{nIdxLayer, nIdxCh}];
  return UpdateStatusChamber(listStatus, numStatusPrev, nLumiCurr, numStatus);
}

Int_t GEMDQMHarvester::UpdateStatusChamber(
    Int_t nIdxLayer, Int_t nIdxCh, Int_t nIdxVFAT, Int_t nLumiCurr, NumStatus numStatus) {
  if (!bIsStatusChambersInit_)
    return 0;
  if (0 >= nIdxCh || nIdxCh > mapNumChPerChamber_[nIdxLayer])
    return 0;
  if (0 >= nIdxVFAT || nIdxVFAT > nNumVFATs_)
    return 0;
  auto &listStatus = mapStatusVFATsSummary_[{nIdxLayer, nIdxCh, nIdxVFAT}];
  auto &numStatusPrev = mapNumStatusVFATsSummary_[{nIdxLayer, nIdxCh, nIdxVFAT}];
  return UpdateStatusChamber(listStatus, numStatusPrev, nLumiCurr, numStatus);
}

Int_t GEMDQMHarvester::UpdateStatusChamber(std::vector<StatusInfo> &listStatus,
                                           NumStatus &numStatus,
                                           Int_t nLumiCurr,
                                           NumStatus numStatusNew) {
  // First of all, the current lumi section will be assessed, of which the result will be stored in nStatus
  Int_t nStatus = 0;

  Float_t fNumAddErrVFAT = numStatusNew.fNumErrVFAT_ - numStatus.fNumErrVFAT_;
  Float_t fNumAddWarnVFAT = numStatusNew.fNumWarnVFAT_ - numStatus.fNumWarnVFAT_;
  Float_t fNumAddErrOH = numStatusNew.fNumErrOH_ - numStatus.fNumErrOH_;
  Float_t fNumAddWarnOH = numStatusNew.fNumWarnOH_ - numStatus.fNumWarnOH_;
  Float_t fNumAddErrAMC = numStatusNew.fNumErrAMC_ - numStatus.fNumErrAMC_;
  Float_t fNumAddWarnAMC = numStatusNew.fNumWarnAMC_ - numStatus.fNumWarnAMC_;
  Float_t fNumAddErrAMC13 = numStatusNew.fNumErrAMC13_ - numStatus.fNumErrAMC13_;

  numStatus.fNumTotal_ = numStatusNew.fNumTotal_;
  numStatus.fNumOcc_ = numStatusNew.fNumOcc_;
  numStatus.fNumErrVFAT_ = numStatusNew.fNumErrVFAT_;
  numStatus.fNumWarnVFAT_ = numStatusNew.fNumWarnVFAT_;
  numStatus.fNumErrOH_ = numStatusNew.fNumErrOH_;
  numStatus.fNumWarnOH_ = numStatusNew.fNumWarnOH_;
  numStatus.fNumErrAMC_ = numStatusNew.fNumErrAMC_;
  numStatus.fNumWarnAMC_ = numStatusNew.fNumWarnAMC_;
  numStatus.fNumErrAMC13_ = numStatusNew.fNumErrAMC13_;

  nStatus = (numStatusNew.fNumOcc_ > 0 ? 1 << nBitOcc_ : 0) | (fNumAddErrAMC13 > 0 ? 1 << nBitErrAMC13_ : 0) |
            (fNumAddErrAMC > 0 ? 1 << nBitErrAMC_ : 0) | (fNumAddWarnAMC > 0 ? 1 << nBitWarnAMC_ : 0) |
            (fNumAddErrOH > 0 ? 1 << nBitErrOH_ : 0) | (fNumAddWarnOH > 0 ? 1 << nBitWarnOH_ : 0) |
            (fNumAddErrVFAT > 0 ? 1 << nBitErrVFAT_ : 0) | (fNumAddWarnVFAT > 0 ? 1 << nBitWarnVFAT_ : 0);

  // Only used in the next if statement; See statusLast
  StatusInfo statusNew;
  statusNew.nLumiStart = nLumiCurr;
  statusNew.nLumiEnd = nLumiCurr;
  statusNew.nStatus = nStatus;

  if (listStatus.empty()) {
    listStatus.push_back(statusNew);
  } else {
    auto &statusLastPre = listStatus.back();
    if (statusLastPre.nStatus == nStatus) {
      statusLastPre.nLumiEnd = nLumiCurr;
    } else {
      listStatus.push_back(statusNew);
    }
  }

  return 0;
}

void GEMDQMHarvester::createLumiFuncHist(edm::Service<DQMStore> &store,
                                         std::string strSuffix,
                                         Int_t nIdxLayer,
                                         Int_t nLumiCurr) {
  auto &nNumCh = mapNumChPerChamber_[nIdxLayer];

  MonitorElement *h2Summary;

  //Int_t nLumiCurrLowRes = ( ( nLumiCurr - 1 ) / nResolutionLumi_ ) * nResolutionLumi_;
  Int_t nNumBinLumi = ((nLumiCurr - 1) / nResolutionLumi_) + 1;
  Int_t nMaxBin = 0;

  // Creating or Summoning the corresponding histogram
  if (mapHistLumiFunc_.find(nIdxLayer) == mapHistLumiFunc_.end()) {
    store->setCurrentFolder(strDirSummary_);
    h2Summary = store->book2S("chamberStatus_inLumi" + strSuffix,
                              "Chamber status on lumi-block " + strSuffix.substr(1),
                              nMaxLumi_ / nResolutionLumi_,
                              1.0,
                              (Float_t)(nMaxLumi_ + 1),
                              //nNumBinLumi, 1.0, (Float_t)( nLumiCurr + 1 ),
                              nNumCh,
                              0.5,
                              nNumCh + 0.5);
    mapHistLumiFunc_[nIdxLayer] = h2Summary;

    h2Summary->setXTitle("Luminosity block");
    h2Summary->setYTitle("Chamber");
    for (Int_t i = 1; i <= nNumCh; i++) {
      h2Summary->setBinLabel(i, Form("%i", i), 2);
    }
  } else {
    h2Summary = mapHistLumiFunc_[nIdxLayer];
  }

  for (Int_t nIdxCh = 1; nIdxCh <= nNumCh; nIdxCh++) {
    auto &listStatus = mapStatusChambersSummary_[{nIdxLayer, nIdxCh}];

    Int_t nIdxStatus = 0;
    for (Int_t nIdxLumi = 0; nIdxLumi < nNumBinLumi; nIdxLumi++) {
      // Lumis covered by these values (nLumiStart <=, <= nLumiEnd) are counted for the current bin
      Int_t nLumiStart = 1 + nIdxLumi * nResolutionLumi_;
      Int_t nLumiEnd = (1 + nIdxLumi) * nResolutionLumi_;
      if (nLumiEnd > nLumiCurr)
        nLumiEnd = nLumiCurr;

      Int_t nStatusSum = 0;
      while (true) {  // No worries, nIdxStatus must increase and reach at listStatus.size()
        // True: It was too past so that
        //       the lumi range of listStatus[ nIdxStatus ] is out of the coverage of the current bin
        if (listStatus[nIdxStatus].nLumiEnd < nLumiStart) {
          nIdxStatus++;
          if (nIdxStatus >= (int)listStatus.size()) {
            break;  // For safety
          }
          continue;
        }

        nStatusSum = listStatus[nIdxStatus].nStatus;

        // True: This is the last item of listStatus which is covered by the current bin
        if (nIdxStatus + 1 >= (int)listStatus.size() || listStatus[nIdxStatus].nLumiEnd >= nLumiEnd) {
          break;
        }

        nIdxStatus++;
        if (nIdxStatus >= (int)listStatus.size()) {
          break;  // For safety
        }
      }

      nStatusSum &= ~(1 << nBitOcc_);  // No need of displaying the digi occupancy
      h2Summary->setBinContent(nIdxLumi + 1, nIdxCh, nStatusSum);
      if (nMaxBin < nIdxLumi + 1)
        nMaxBin = nIdxLumi + 1;
    }
  }

  for (Int_t nX = 1; nX <= nMaxBin; nX++) {
    h2Summary->setBinContent(nX, 0, 1);
  }
}

std::string getNameChamberOccGE11(std::string strSuffix, Int_t nIdxCh) {
  char cRegion;
  char cChType = (nIdxCh % 2 == 0 ? 'L' : 'S');
  Int_t nLayer;

  if (strSuffix.find("-M-") != std::string::npos)
    cRegion = 'M';
  else if (strSuffix.find("-P-") != std::string::npos)
    cRegion = 'P';
  else
    return "";

  if (strSuffix.find("-L1") != std::string::npos)
    nLayer = 1;
  else if (strSuffix.find("-L2") != std::string::npos)
    nLayer = 2;
  else
    return "";

  return Form(
      "GEM/Digis/occupancy_GE11-%c-L%i/occ_GE11-%c-%02iL%i-%c", cRegion, nLayer, cRegion, nIdxCh, nLayer, cChType);
}

std::string getNameChamberOccGE21(std::string strSuffix, Int_t nIdxChamber) {
  return "";  // FIXME
}

std::string getNameChamberOccNull(std::string strSuffix, Int_t nIdxChamber) {
  return "";  // For an initialization
}

void GEMDQMHarvester::createInactiveChannelFracHist(edm::Service<DQMStore> &store,
                                                    std::string strSuffix,
                                                    Int_t nNumChamber) {
  std::string strTitle = "The fraction of inactive channels in " + strSuffix.substr(1);
  MonitorElement *h2InactiveChannel =
      store->book1D("inactive_frac_chamber" + strSuffix, strTitle, nNumChamber, 0.5, nNumChamber + 0.5);
  h2InactiveChannel->setXTitle("Chamber");
  h2InactiveChannel->setYTitle("Fraction of inactive channels");
  for (Int_t i = 1; i <= nNumChamber; i++) {
    h2InactiveChannel->setBinLabel(i, Form("%i", i), 1);
  }

  std::string (*funcNameCh)(std::string, Int_t) = getNameChamberOccNull;

  if (strSuffix.find("_GE11") != std::string::npos) {
    funcNameCh = getNameChamberOccGE11;
  } else if (strSuffix.find("_GE21") != std::string::npos) {
    funcNameCh = getNameChamberOccGE21;
  }

  for (Int_t nIdxCh = 1; nIdxCh <= nNumChamber; nIdxCh++) {
    std::string strNameCh = funcNameCh(strSuffix, nIdxCh);
    MonitorElement *h2SrcChamberOcc = store->get(strNameCh);
    if (h2SrcChamberOcc == nullptr) {
      // FIXME: It's about sending a message
      continue;
    }

    Int_t nNumBinX = h2SrcChamberOcc->getNbinsX();
    Int_t nNumBinY = h2SrcChamberOcc->getNbinsY();
    Int_t nNumAllChannel = nNumBinX * nNumBinY;
    auto *histData = h2SrcChamberOcc->getTH2F();
    auto *pdData = histData->GetArray();
    Int_t nNumChannelInactive = 0;
    for (Int_t j = 1; j <= nNumBinY; j++)
      for (Int_t i = 1; i <= nNumBinX; i++) {
        if (pdData[j * (nNumBinX + 2) + i] <= 0) {
          nNumChannelInactive++;
        }
      }
    h2InactiveChannel->setBinContent(nIdxCh, ((Double_t)nNumChannelInactive) / nNumAllChannel);
  }
}

DEFINE_FWK_MODULE(GEMDQMHarvester);
