#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "EventFilter/GEMRawToDigi/interface/GEMVfatStatusDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMGEBdataCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMAMCdataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include <string>
#include <fstream>

#include <TFile.h>
#include <TDirectoryFile.h>

//----------------------------------------------------------------------------------------------------

typedef struct tagTimeStoreItem {
  std::string strName;
  std::string strTitle;
  std::string strAxisX;

  MonitorElement *h2Histo;

  Int_t nNbinY;
  Int_t nNbinMin;
  Int_t nNbinMax;

  std::vector<std::vector<Double_t>> listOccupy;
  std::vector<std::vector<Double_t>> listRecord;
} TimeStoreItem;

class GEMDQMStatusDigi : public DQMEDAnalyzer {
public:
  GEMDQMStatusDigi(const edm::ParameterSet &cfg);
  ~GEMDQMStatusDigi() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookHistogramsChamberPart(DQMStore::IBooker &, GEMDetId &);
  void bookHistogramsStationPart(DQMStore::IBooker &, GEMDetId &);
  void bookHistogramsAMCPart(DQMStore::IBooker &);
  void bookHistogramsTimeRecordPart(DQMStore::IBooker &);

  int SetInfoChambers();
  int SetConfigTimeRecord();
  int LoadPrevData();

  Int_t seekIdx(std::vector<GEMDetId> &listLayers, UInt_t unId);

  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;
  void endRun(edm::Run const &run, edm::EventSetup const &eSetup) override;

private:
  const GEMGeometry *initGeometry(edm::EventSetup const &iSetup);

  void AddLabel();

  std::string suffixChamber(GEMDetId &id);
  std::string suffixLayer(GEMDetId &id);

  void FillBits(MonitorElement *monitor, uint64_t unVal, int nNumBits);
  void FillBits(MonitorElement *monitor, uint64_t unVal, int nNumBits, int nY);

  const GEMGeometry *GEMGeometry_;
  std::vector<GEMChamber> gemChambers_;

  int nVfat_ = 24;

  int nNCh_;

  int cBit_ = 9;
  int qVFATBit_ = 5;
  int fVFATBit_ = 4;
  int eBit_ = 16;
  int amcStatusBit_ = 6;

  int nNEvtPerSec_;
  int nNSecPerBin_;
  int nNTimeBinTotal_;
  int nNTimeBinPrimitive_;

  int nIdxFirstStrip_;

  int nNBxBin_;
  int nNBxRange_;

  std::string strFmtSummaryLabel_;
  bool bFlipSummary_;
  bool bPerSuperchamber_;

  std::string strPathPrevDQMRoot_;

  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagGEB_;
  edm::EDGetToken tagAMC_;
  edm::EDGetToken tagDigi_;

  std::vector<Int_t> listAMCSlots_;

  std::vector<GEMDetId> m_listLayers;
  std::vector<GEMDetId> m_listChambers;

  MonitorElement *h1_vfat_qualityflag_;
  MonitorElement *h2_vfat_qualityflag_;

  std::unordered_map<UInt_t, MonitorElement *> listVFATQualityFlag_;
  std::unordered_map<UInt_t, MonitorElement *> listVFATBC_;
  std::unordered_map<UInt_t, MonitorElement *> listVFATEC_;

  std::unordered_map<UInt_t, MonitorElement *> listGEBInputStatus_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBInputID_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBVFATWordCnt_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBVFATWordCntT_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBZeroSupWordsCnt_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBbcOH_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBecOH_;
  std::unordered_map<UInt_t, MonitorElement *> listGEBOHCRC_;

  std::unordered_map<UInt_t, Bool_t> m_mapStatusFill;
  std::unordered_map<UInt_t, Bool_t> m_mapStatusErr;

  MonitorElement *h1_amc_ttsState_;
  MonitorElement *h1_amc_davCnt_;
  MonitorElement *h1_amc_buffState_;
  MonitorElement *h1_amc_oosGlib_;
  MonitorElement *h1_amc_chTimeOut_;
  MonitorElement *h2AMCStatus_;

  MonitorElement *m_summaryReport_;
  MonitorElement *h2SummaryStatus_;

  // For more information, see SetConfigTimeRecord()
  std::unordered_map<UInt_t, TimeStoreItem> listTimeStore_;
  Int_t nStackedEvt_;
};

const GEMGeometry *GEMDQMStatusDigi::initGeometry(edm::EventSetup const &iSetup) {
  const GEMGeometry *GEMGeometry_ = nullptr;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry> &e) {
    edm::LogError("MuonGEMBaseValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return nullptr;
  }
  return GEMGeometry_;
}

using namespace std;
using namespace edm;

GEMDQMStatusDigi::GEMDQMStatusDigi(const edm::ParameterSet &cfg) {
  tagVFAT_ = consumes<GEMVfatStatusDigiCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel"));
  tagGEB_ = consumes<GEMGEBdataCollection>(cfg.getParameter<edm::InputTag>("GEBInputLabel"));
  tagAMC_ = consumes<GEMAMCdataCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel"));
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));

  listAMCSlots_ = cfg.getParameter<std::vector<int>>("AMCSlots");

  strFmtSummaryLabel_ = cfg.getParameter<std::string>("summaryLabelFmt");
  bFlipSummary_ = cfg.getParameter<bool>("flipSummary");
  bPerSuperchamber_ = cfg.getParameter<bool>("perSuperchamber");

  strPathPrevDQMRoot_ = cfg.getParameter<std::string>("pathOfPrevDQMRoot");
  nNEvtPerSec_ = cfg.getParameter<int>("numOfEvtPerSec");
  nNSecPerBin_ = cfg.getParameter<int>("secOfEvtPerBin");
  nNTimeBinPrimitive_ = cfg.getParameter<int>("totalTimeInterval");

  nIdxFirstStrip_ = cfg.getParameter<int>("idxFirstStrip");

  nNBxRange_ = cfg.getParameter<int>("bxRange");
  nNBxBin_ = cfg.getParameter<int>("bxBin");
}

void GEMDQMStatusDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "vfatStatus"));
  desc.add<edm::InputTag>("GEBInputLabel", edm::InputTag("muonGEMDigis", "gebStatus"));
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCdata"));
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));

  std::vector<int> listAMCSlotsDef = {0, 1, 2, 3, 4, 5, 6, 7};
  desc.add<std::vector<int>>("AMCSlots", listAMCSlotsDef);  // TODO: Find how to get this from the geometry

  desc.add<std::string>("summaryLabelFmt", "GE%(station_signed)+i/%(layer)i");
  desc.add<bool>("flipSummary", false);
  desc.add<bool>("perSuperchamber", true);

  desc.add<std::string>("pathOfPrevDQMRoot", "");
  desc.add<int>("numOfEvtPerSec", 100);
  desc.add<int>("secOfEvtPerBin", 10);
  desc.add<int>("totalTimeInterval", 50000);

  desc.add<int>("idxFirstStrip", 0);

  desc.add<int>("bxRange", 10);
  desc.add<int>("bxBin", 20);

  descriptions.add("GEMDQMStatusDigi", desc);
}

std::string GEMDQMStatusDigi::suffixChamber(GEMDetId &id) {
  return "Gemini_" + to_string(id.chamber()) + "_GE" + (id.region() > 0 ? "p" : "m") + to_string(id.station()) + "_" +
         to_string(id.layer());
}

std::string GEMDQMStatusDigi::suffixLayer(GEMDetId &id) {
  return std::string("st_") + (id.region() >= 0 ? "p" : "m") + std::to_string(id.station()) +
         (bPerSuperchamber_ ? "_la_" + std::to_string(id.layer()) : "");
}

int GEMDQMStatusDigi::SetInfoChambers() {
  const std::vector<const GEMSuperChamber *> &superChambers_ = GEMGeometry_->superChambers();
  for (auto sch : superChambers_) {
    int nLayer = sch->nChambers();
    for (int l = 0; l < nLayer; l++) {
      Bool_t bExist = false;
      for (auto ch : gemChambers_)
        if (ch.id() == sch->chamber(l + 1)->id())
          bExist = true;
      if (bExist)
        continue;

      gemChambers_.push_back(*sch->chamber(l + 1));
    }
  }

  // End: Loading the GEM geometry

  // Start: Set the configurations

  m_listLayers.clear();

  // Summarizing geometry configurations
  for (auto ch : gemChambers_) {
    GEMDetId gid = ch.id();

    GEMDetId layerID(gid.region(), gid.ring(), gid.station(), (bPerSuperchamber_ ? gid.layer() : 0), 0, 0);
    Bool_t bOcc = false;

    for (auto lid : m_listLayers) {
      if (lid == layerID) {
        bOcc = true;
        break;
      }
    }

    if (!bOcc)
      m_listLayers.push_back(layerID);

    GEMDetId chamberID(0, 1, 1, (bPerSuperchamber_ ? 0 : gid.layer()), gid.chamber(), 0);
    bOcc = false;

    for (auto cid : m_listChambers) {
      if (cid == chamberID) {
        bOcc = true;
        break;
      }
    }

    if (!bOcc)
      m_listChambers.push_back(chamberID);
  }

  // Preliminary for sorting the summaries
  auto lambdaLayer = [this](GEMDetId a, GEMDetId b) -> Bool_t {
    Int_t nFlipSign = (this->bFlipSummary_ ? -1 : 1);
    Int_t nA = nFlipSign * a.region() * (20 * a.station() + a.layer());
    Int_t nB = nFlipSign * b.region() * (20 * b.station() + b.layer());
    return nA > nB;
  };

  auto lambdaChamber = [this](GEMDetId a, GEMDetId b) -> Bool_t {
    Int_t nA = 20 * a.chamber() + a.layer();
    Int_t nB = 20 * b.chamber() + b.layer();
    return nA < nB;
  };

  // Sorting the summaries
  std::sort(m_listLayers.begin(), m_listLayers.end(), lambdaLayer);
  std::sort(m_listChambers.begin(), m_listChambers.end(), lambdaChamber);

  nNCh_ = (int)m_listChambers.size();

  return 0;
}

// 0: General; for whole AMC slots
int GEMDQMStatusDigi::SetConfigTimeRecord() {
  TimeStoreItem newTimeStore;

  newTimeStore.nNbinMin = 0;
  newTimeStore.nNbinMax = 0;

  std::string strCommonName = "per_time_";

  // Very general GEMDetId
  newTimeStore.strName = strCommonName + "status_AMCslots";
  newTimeStore.strTitle = "Status of AMC slots per time";
  newTimeStore.strAxisX = "AMC slot";
  newTimeStore.nNbinY = listAMCSlots_.size();
  listTimeStore_[0] = newTimeStore;

  for (auto layerId : m_listLayers) {
    std::string strSuffix =
        (layerId.region() > 0 ? "p" : "m") + std::to_string(layerId.station()) + "_" + std::to_string(layerId.layer());

    newTimeStore.strName = strCommonName + "status_GEB_" + suffixLayer(layerId);
    newTimeStore.strTitle = "";
    newTimeStore.strAxisX = "Chamber";

    newTimeStore.nNbinY = nNCh_;
    listTimeStore_[layerId] = newTimeStore;
  }

  for (auto ch : gemChambers_) {
    auto chId = ch.id();
    GEMDetId chIdStatus(chId.region(), chId.ring(), chId.station(), chId.layer(), chId.chamber(), 1);
    GEMDetId chIdDigi(chId.region(), chId.ring(), chId.station(), chId.layer(), chId.chamber(), 2);
    GEMDetId chIdBx(chId.region(), chId.ring(), chId.station(), chId.layer(), chId.chamber(), 3);

    std::string strSuffix = suffixChamber(chId);

    newTimeStore.strName = strCommonName + "status_chamber_" + strSuffix;
    newTimeStore.strTitle = "";
    newTimeStore.strAxisX = "VFAT";

    newTimeStore.nNbinY = nVfat_;
    listTimeStore_[chIdStatus] = newTimeStore;

    newTimeStore.strName = strCommonName + "digi_chamber_" + strSuffix;
    newTimeStore.strTitle = "";
    newTimeStore.strAxisX = "VFAT";

    newTimeStore.nNbinY = nVfat_;
    listTimeStore_[chIdDigi] = newTimeStore;

    newTimeStore.strName = strCommonName + "bx_chamber_" + strSuffix;
    newTimeStore.strTitle = "";
    newTimeStore.strAxisX = "Bunch crossing";

    newTimeStore.nNbinY = nNBxBin_;
    newTimeStore.nNbinMin = -nNBxRange_;
    newTimeStore.nNbinMax = nNBxRange_;

    listTimeStore_[chIdBx] = newTimeStore;

    newTimeStore.nNbinMin = newTimeStore.nNbinMax = 0;
  }

  return 0;
}

int GEMDQMStatusDigi::LoadPrevData() {
  TFile *fPrev;
  Bool_t bSync = true;

  nStackedEvt_ = 0;

  for (auto &itStore : listTimeStore_) {
    itStore.second.listOccupy.clear();
    itStore.second.listRecord.clear();
  }

  if (strPathPrevDQMRoot_ == "")
    return 0;

  std::ifstream fExist(strPathPrevDQMRoot_.c_str());
  if (!fExist.good())
    return 0;
  fExist.close();

  fPrev = new TFile(strPathPrevDQMRoot_.c_str());
  if (fPrev == NULL)
    return 1;

  std::cout << strPathPrevDQMRoot_ << " is being loaded" << std::endl;
  std::string strRunnum = ((TDirectoryFile *)fPrev->Get("DQMData"))->GetListOfKeys()->At(0)->GetName();

  for (auto &itStore : listTimeStore_) {
    TH2F *h2Prev =
        (TH2F *)fPrev->Get(("DQMData/" + strRunnum + "/GEM/Run summary/StatusDigi/" + itStore.second.strName).c_str());
    auto &listCurrRec = itStore.second.listRecord;
    auto &listCurrOcc = itStore.second.listOccupy;

    listCurrRec.clear();
    listCurrOcc.clear();

    if (h2Prev == NULL)
      continue;  // Hmmm, is it an error...?

    Int_t nNbinX = h2Prev->GetNbinsX();
    Int_t nNbinY = h2Prev->GetNbinsY();

    Int_t nX, nY;
    Int_t nXMaxOcc = 0;
    Int_t nNNonFullFill = 0;

    for (nX = 1; nX <= nNbinX; nX++) {
      Int_t nNEvtCurrTime = h2Prev->GetBinContent(nX, 0);  // The (nX, 0)-bin keeps the number of filled events
      if (0 < nNEvtCurrTime && nNEvtCurrTime < nNSecPerBin_ * nNEvtPerSec_)
        nNNonFullFill++;
      if (nNEvtCurrTime > 0)
        nXMaxOcc = nX;
    }

    if (nNNonFullFill > 1) {  // The last bin may not full
      std::cerr << "WARNING: There is a bin of the previous histograms which is not fully filled" << std::endl;
    }

    for (nY = 1; nY <= nNbinY; nY++) {
      listCurrRec.emplace_back();
      listCurrOcc.emplace_back();

      if (nXMaxOcc > 0) {
        for (nX = 1; nX <= nXMaxOcc; nX++) {
          //listCurr[ nY - 1 ].push_back(h2Prev->GetBinContent(nX, nY));
          listCurrRec[nY - 1].push_back(0);  // If hadd is used, then it must be blanked
          listCurrOcc[nY - 1].push_back(0);  // If hadd is used, then it must be blanked
        }

        if (nNNonFullFill <= 0) {
          listCurrRec[nY - 1].push_back(0);
          listCurrOcc[nY - 1].push_back(0);
        }
      } else {  // Empty...?
        listCurrRec[nY - 1].push_back(0);
        listCurrOcc[nY - 1].push_back(0);
      }
    }

    if (nXMaxOcc <= 0)
      std::cout << "nXMacOcc has a problem" << std::endl;
    if (nXMaxOcc <= 0)
      continue;  // Empty...?

    // Keeping the number of lastly stacked events
    // And also checking if the sync is okay
    if (nStackedEvt_ == 0)
      nStackedEvt_ = h2Prev->GetBinContent(nXMaxOcc, 0);
    else {
      if (nStackedEvt_ != h2Prev->GetBinContent(nXMaxOcc, 0))
        bSync = false;
    }
  }

  if (nStackedEvt_ == nNSecPerBin_ * nNEvtPerSec_)
    nStackedEvt_ = 0;

  if (!bSync) {  // No sync...!
    std::cerr << "WARNING: No sync on time histograms" << std::endl;
  }

  fPrev->Close();

  return 0;
}

void GEMDQMStatusDigi::bookHistogramsChamberPart(DQMStore::IBooker &ibooker, GEMDetId &gid) {
  std::string hName, hTitle;

  UInt_t unBinPos;
  //std::cout << "B: " << gid << std::endl;

  std::string strIdxName = suffixChamber(gid);
  std::string strIdxTitle = "GEMINIm" + to_string(gid.chamber()) + " in GE" + (gid.region() > 0 ? "+" : "-") +
                            to_string(gid.station()) + "/" + to_string(gid.layer());

  hName = "vfatStatus_QualityFlag_" + strIdxName;
  hTitle = "VFAT quality " + strIdxTitle;
  hTitle += ";VFAT;";
  std::cout << "booking: " << gid << std::endl;
  listVFATQualityFlag_[gid] = ibooker.book2D(hName, hTitle, nVfat_, 0, nVfat_, 9, 0, 9);

  hName = "vfatStatus_BC_" + strIdxName;
  hTitle = "VFAT bunch crossing " + strIdxTitle;
  hTitle += ";Bunch crossing;VFAT";
  listVFATBC_[gid] = ibooker.book2D(hName, hTitle, nNBxBin_, -nNBxRange_, nNBxRange_, nVfat_, 0, nVfat_);

  hName = "vfatStatus_EC_" + strIdxName;
  hTitle = "VFAT event counter " + strIdxTitle;
  hTitle += ";Event counter;VFAT";
  listVFATEC_[gid] = ibooker.book2D(hName, hTitle, 256, 0, 256, nVfat_, 0, nVfat_);

  unBinPos = 1;
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "Good", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "CRC fail", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "b1010 fail", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "b1100 fail", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "b1110 fail", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "Hamming error", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "AFULL", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "SEUlogic", 2);
  listVFATQualityFlag_[gid]->setBinLabel(unBinPos++, "SUEI2C", 2);

  m_mapStatusFill[gid] = false;
  m_mapStatusErr[gid] = false;
}

void GEMDQMStatusDigi::bookHistogramsStationPart(DQMStore::IBooker &ibooker, GEMDetId &lid) {
  UInt_t unBinPos;

  Int_t re = lid.region();
  UInt_t st = lid.station();
  UInt_t la = lid.layer();

  auto newbookGEB = [this](DQMStore::IBooker &ibooker,
                           std::string strName,
                           std::string strTitle,
                           std::string strAxis,
                           GEMDetId &lid,
                           int nLayer,
                           int nStation,
                           int re,
                           int nBin,
                           float fMin,
                           float fMax) -> MonitorElement * {
    strName = strName + "_" + suffixLayer(lid);
    strTitle = strTitle + ", station: " + (re >= 0 ? "+" : "-") + std::to_string(nStation);

    if (bPerSuperchamber_) {
      strTitle += ", layer: " + std::to_string(nLayer);
    }

    strTitle += ";Chamber;" + strAxis;

    auto hNew = ibooker.book2D(strName, strTitle, this->nNCh_, 0, this->nNCh_, nBin, fMin, fMax);

    for (Int_t i = 0; i < this->nNCh_; i++) {
      auto &gid = this->m_listChambers[i];
      Int_t nCh = gid.chamber() + (this->bPerSuperchamber_ ? 0 : gid.layer() - 1);
      hNew->setBinLabel(i + 1, std::to_string(nCh), 1);
    }

    return hNew;
  };

  listGEBInputStatus_[lid] =
      newbookGEB(ibooker, "geb_input_status", "inputStatus", "", lid, la, st, re, eBit_, 0, eBit_);
  listGEBInputID_[lid] = newbookGEB(ibooker, "geb_input_ID", "inputID", "Input ID", lid, la, st, re, 32, 0, 32);
  listGEBVFATWordCnt_[lid] =
      newbookGEB(ibooker, "geb_no_vfats", "nvfats in header", "Number of VFATs in header", lid, la, st, re, 25, 0, 25);
  listGEBVFATWordCntT_[lid] = newbookGEB(
      ibooker, "geb_no_vfatsT", "nvfats in trailer", "Number of VFATs in trailer", lid, la, st, re, 25, 0, 25);
  listGEBZeroSupWordsCnt_[lid] = newbookGEB(
      ibooker, "geb_zeroSupWordsCnt", "zeroSupWordsCnt", "Zero sup. words count", lid, la, st, re, 10, 0, 10);

  listGEBbcOH_[lid] =
      newbookGEB(ibooker, "geb_bcOH", "OH bunch crossing", "OH bunch crossing", lid, la, st, re, 3600, 0, 3600);
  listGEBecOH_[lid] =
      newbookGEB(ibooker, "geb_ecOH", "OH event coounter", "OH event counter", lid, la, st, re, 256, 0, 256);
  listGEBOHCRC_[lid] =
      newbookGEB(ibooker, "geb_OHCRC", "CRC of OH data", "CRC of OH data", lid, la, st, re, 65536, 0, 65536);

  unBinPos = 1;
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "BX mismatch GLIB OH", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "BX mismatch GLIB VFAT", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "OOS GLIB OH", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "OOS GLIB VFAT", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "No VFAT marker", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "Event size warn", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "L1AFIFO near full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "InFIFO near full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "EvtFIFO near full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "Event size overflow", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "L1AFIFO full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "InFIFO full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "EvtFIFO full", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "Input FIFO underflow", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "Stuck data", 2);
  listGEBInputStatus_[lid]->setBinLabel(unBinPos++, "Event FIFO underflow", 2);
}

void GEMDQMStatusDigi::bookHistogramsAMCPart(DQMStore::IBooker &ibooker) {
  h2AMCStatus_ = ibooker.book2D("amc_statusflag",
                                "Status of AMC slots;AMC slot;",
                                listAMCSlots_.size(),
                                0,
                                listAMCSlots_.size(),
                                amcStatusBit_,
                                0,
                                amcStatusBit_);

  uint32_t unBinPos = 1;
  h2AMCStatus_->setBinLabel(unBinPos++, "BC0 not locked", 2);
  h2AMCStatus_->setBinLabel(unBinPos++, "DAQ not ready", 2);
  h2AMCStatus_->setBinLabel(unBinPos++, "DAQ clock not locked", 2);
  h2AMCStatus_->setBinLabel(unBinPos++, "MMCM not locked", 2);
  h2AMCStatus_->setBinLabel(unBinPos++, "Back pressure", 2);
  h2AMCStatus_->setBinLabel(unBinPos++, "GLIB out-of-sync", 2);
}

void GEMDQMStatusDigi::bookHistogramsTimeRecordPart(DQMStore::IBooker &ibooker) {
  //Int_t nNBinX = ( listTimeStore_[ 0 ].listRecord.size() / nNTimeBinTotal_ + 1 ) * nNTimeBinTotal_;

  for (auto &itStore : listTimeStore_) {
    auto &infoCurr = itStore.second;

    Float_t fMin = -0.5, fMax = infoCurr.nNbinY - 0.5;

    if (infoCurr.nNbinMin < infoCurr.nNbinMax) {
      fMin = infoCurr.nNbinMin;
      fMax = infoCurr.nNbinMax;
    }

    infoCurr.h2Histo = ibooker.book2D(
        infoCurr.strName,
        //infoCurr.strTitle + ";Time (per " + std::to_string(nNSecPerBin_) + " sec);" + infoCurr.strAxisX,
        infoCurr.strTitle + ";Per " + std::to_string(nNSecPerBin_ * nNEvtPerSec_) + " events;" + infoCurr.strAxisX,
        nNTimeBinPrimitive_,
        0,
        nNTimeBinPrimitive_,
        infoCurr.nNbinY,
        fMin,
        fMax);

    if (seekIdx(m_listLayers, itStore.first) >= 0) {
      for (Int_t i = 0; i < nNCh_; i++) {
        auto &gid = m_listChambers[i];
        Int_t nCh = gid.chamber() + (bPerSuperchamber_ ? 0 : gid.layer() - 1);
        infoCurr.h2Histo->setBinLabel(i + 1, std::to_string(nCh), 2);
      }
    }
  }
}

// To make labels like python, with std::(unordered_)map
std::string printfWithMap(std::string strFmt, std::unordered_map<std::string, Int_t> mapArg) {
  std::string strRes = strFmt;
  char szOutFmt[64];
  size_t unPos, unPosEnd;

  for (unPos = strRes.find('%'); unPos != std::string::npos; unPos = strRes.find('%', unPos + 1)) {
    unPosEnd = strRes.find(')', unPos);
    if (strRes[unPos + 1] != '(' || unPosEnd == std::string::npos)
      break;  // Syntax error

    // Extracting the key
    std::string strKey = strRes.substr(unPos + 2, unPosEnd - (unPos + 2));

    // To treat formats like '%5i' or '%02i', extracting '5' or '02'
    // After do this,
    std::string strOptNum = "%";
    unPosEnd++;

    for (;; unPosEnd++) {
      if (!('0' <= strRes[unPosEnd] && strRes[unPosEnd] <= '9') && strRes[unPosEnd] != '+')
        break;
      strOptNum += strRes[unPosEnd];
    }

    if (strRes[unPosEnd] != 'i' && strRes[unPosEnd] != 'd')
      break;  // Syntax error
    strOptNum += strRes[unPosEnd];
    unPosEnd++;

    sprintf(szOutFmt, strOptNum.c_str(), mapArg[strKey]);
    strRes = strRes.substr(0, unPos) + szOutFmt + strRes.substr(unPosEnd);
  }

  if (unPos != std::string::npos) {  // It means... an syntax error occurs!
    std::cerr << "ERROR: Syntax error on printfWithMap(); " << std::endl;
    return "";
  }

  return strRes;
}

void GEMDQMStatusDigi::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &iSetup) {
  // Start: Loading the GEM geometry

  GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  SetInfoChambers();

  // Setting the informations for time histograms
  SetConfigTimeRecord();
  LoadPrevData();

  // The loading does not work well or something goes wrong...
  // In this case, we need to make a new start of recording.
  for (auto &itStore : listTimeStore_) {
    auto &infoCurr = itStore.second;

    if (infoCurr.nNbinY != (Int_t)infoCurr.listRecord.size()) {
      if (infoCurr.listRecord.size() > 0) {
        std::cerr << "WARNING: The previous result is not compatible to the current geometry. "
                  << "We will discard the previous data." << std::endl;

        infoCurr.listRecord.clear();
        infoCurr.listOccupy.clear();
      }

      for (Int_t i = 0; i < infoCurr.nNbinY; i++) {
        infoCurr.listRecord.emplace_back();
        infoCurr.listRecord[i].push_back(0);

        infoCurr.listOccupy.emplace_back();
        infoCurr.listOccupy[i].push_back(0);
      }
    }
    /*for ( Int_t i = 0 ; i < (Int_t)infoCurr.listRecord[ 0 ].size() ; i++ ) 
    for ( Int_t j = 0 ; j < infoCurr.nNbinY ; j++ ) 
      std::cout << "Init time: " << infoCurr.strName << ", " << j << ", " << i << ", " << infoCurr.listRecord[ j ][ i ] << std::endl;
    
    std::cout << "Init time: " << itStore.first << ", " << infoCurr.strName << ", " << infoCurr.nNbinY << ", " << infoCurr.listRecord.size() << "; " << listTimeStore_[ itStore.first ].listRecord.size() << std::endl;*/
  }

  // End: Set the configurations

  // Start: Setting books

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/StatusDigi");

  for (auto ch : gemChambers_) {
    GEMDetId gid = ch.id();
    bookHistogramsChamberPart(ibooker, gid);
  }

  for (auto lid : m_listLayers) {
    bookHistogramsStationPart(ibooker, lid);
  }

  bookHistogramsAMCPart(ibooker);
  bookHistogramsTimeRecordPart(ibooker);

  h1_vfat_qualityflag_ = ibooker.book1D("vfat_quality_flag", "quality and flag", 9, 0, 9);
  h2_vfat_qualityflag_ = ibooker.book2D("vfat_quality_flag_per_geb", "quality and flag", nNCh_, 0, nNCh_, 9, 0, 9);

  h1_amc_ttsState_ = ibooker.book1D("amc_ttsState", "ttsState", 10, 0, 10);
  h1_amc_davCnt_ = ibooker.book1D("amc_davCnt", "davCnt", 10, 0, 10);
  h1_amc_buffState_ = ibooker.book1D("amc_buffState", "buffState", 10, 0, 10);
  h1_amc_oosGlib_ = ibooker.book1D("amc_oosGlib", "oosGlib", 10, 0, 10);
  h1_amc_chTimeOut_ = ibooker.book1D("amc_chTimeOut", "chTimeOut", 10, 0, 10);

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/EventInfo");

  // TODO: We need a study for setting the rule for this
  m_summaryReport_ = ibooker.bookFloat("reportSummary");
  m_summaryReport_->Fill(1.0);

  h2SummaryStatus_ =
      ibooker.book2D("reportSummaryMap", ";Chamber;", nNCh_, 0, nNCh_, m_listLayers.size(), 0, m_listLayers.size());

  for (Int_t i = 0; i < nNCh_; i++) {
    auto &gid = this->m_listChambers[i];
    Int_t nCh = gid.chamber() + (bPerSuperchamber_ ? 0 : gid.layer() - 1);
    h2SummaryStatus_->setBinLabel(i + 1, std::to_string(nCh), 1);
  }

  Int_t nIdxLayer = 0;
  std::unordered_map<std::string, Int_t> mapArg;

  // Start: Labeling section

  for (auto lid : m_listLayers) {
    mapArg["station_signed"] = lid.region() * lid.station();
    mapArg["region"] = lid.region();
    mapArg["station"] = lid.station();
    mapArg["layer"] = lid.layer();
    mapArg["chamber"] = lid.chamber();

    h2SummaryStatus_->setBinLabel(nIdxLayer + 1, printfWithMap(strFmtSummaryLabel_, mapArg), 2);
    nIdxLayer++;
  }
}

void GEMDQMStatusDigi::FillBits(MonitorElement *monitor, uint64_t unVal, int nNumBits) {
  int i = 0;
  uint64_t unFlag = 1;

  for (; i < nNumBits; i++, unFlag <<= 1) {
    if ((unVal & unFlag) != 0) {
      monitor->Fill(i);
    }
  }
}

void GEMDQMStatusDigi::FillBits(MonitorElement *monitor, uint64_t unVal, int nNumBits, int nX) {
  int i = 0;
  uint64_t unFlag = 1;

  for (; i < nNumBits; i++, unFlag <<= 1) {
    if ((unVal & unFlag) != 0) {
      monitor->Fill(nX, i);
    }
  }
}

Int_t GEMDQMStatusDigi::seekIdx(std::vector<GEMDetId> &listLayers, UInt_t unId) {
  if (unId < 256)
    return -1;

  GEMDetId id(unId);
  for (Int_t nIdx = 0; nIdx < (Int_t)listLayers.size(); nIdx++)
    if (id == listLayers[nIdx])
      return nIdx;
  return -1;
}

void GEMDQMStatusDigi::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  edm::Handle<GEMVfatStatusDigiCollection> gemVFAT;
  edm::Handle<GEMGEBdataCollection> gemGEB;
  edm::Handle<GEMAMCdataCollection> gemAMC;
  edm::Handle<GEMDigiCollection> gemDigis;

  event.getByToken(tagVFAT_, gemVFAT);
  event.getByToken(tagGEB_, gemGEB);
  event.getByToken(tagAMC_, gemAMC);
  event.getByToken(tagDigi_, gemDigis);

  auto fillTimeHisto = [](TimeStoreItem &listCurr, int nIdx, bool bFill) -> void {
    return;
    auto &listRecord = listCurr.listRecord[nIdx];
    auto &listOccupy = listCurr.listOccupy[nIdx];
    int nIdxTime = listRecord.size() - 1;

    listOccupy[nIdxTime] = 1;
    if (bFill)
      listRecord[nIdxTime]++;
  };

  for (GEMVfatStatusDigiCollection::DigiRangeIterator vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt) {
    GEMDetId gemid = (*vfatIt).first;
    GEMDetId gemchId = gemid.chamberId();
    //std::cout << "A: " << gemchId << std::endl;
    GEMDetId gemOnlychId(0, 1, 1, (bPerSuperchamber_ ? 0 : gemid.layer()), gemid.chamber(), 0);

    int nIdx = seekIdx(m_listChambers, gemOnlychId);
    int nRoll = gemid.roll();
    const GEMVfatStatusDigiCollection::Range &range = (*vfatIt).second;

    GEMDetId chIdStatus(gemid.region(), gemid.ring(), gemid.station(), gemid.layer(), gemid.chamber(), 1);
    auto &listCurr = listTimeStore_[chIdStatus];

    for (auto vfatStat = range.first; vfatStat != range.second; ++vfatStat) {
      uint64_t unQFVFAT = vfatStat->quality() | (vfatStat->flag() << qVFATBit_);
      if ((unQFVFAT & ~0x1) == 0) {
        unQFVFAT |= 0x1;  // If no error, then it should be 'Good'
      } else {            // Error!!
        m_mapStatusErr[gemchId] = true;
      }

      FillBits(h1_vfat_qualityflag_, unQFVFAT, qVFATBit_ + fVFATBit_);
      FillBits(h2_vfat_qualityflag_, unQFVFAT, qVFATBit_ + fVFATBit_, nIdx);

      int nVFAT = (8 - nRoll) + 8 * vfatStat->phi();  // vfatStat.phi()? or 2 - vfatStat.phi()?

      FillBits(listVFATQualityFlag_[gemchId], unQFVFAT, qVFATBit_ + fVFATBit_, nVFAT);
      //listVFATBC_[ gemchId ]->Fill(vfatStat->bc(), nVFAT);
      listVFATEC_[gemchId]->Fill(vfatStat->ec(), nVFAT);

      fillTimeHisto(listCurr, nVFAT, (unQFVFAT & ~0x1) != 0);
    }
  }

  for (GEMGEBdataCollection::DigiRangeIterator gebIt = gemGEB->begin(); gebIt != gemGEB->end(); ++gebIt) {
    GEMDetId gemid = (*gebIt).first;
    GEMDetId lid(gemid.region(), gemid.ring(), gemid.station(), (bPerSuperchamber_ ? gemid.layer() : 0), 0, 0);
    GEMDetId chid(0, 1, 1, (bPerSuperchamber_ ? 0 : gemid.layer()), gemid.chamber(), 0);

    Int_t nCh = seekIdx(m_listChambers, chid);
    auto &listCurr = listTimeStore_[lid];

    const GEMGEBdataCollection::Range &range = (*gebIt).second;
    for (auto GEBStatus = range.first; GEBStatus != range.second; ++GEBStatus) {
      uint64_t unBit = 0;
      uint64_t unStatus = 0;

      unStatus |= (GEBStatus->bxmVvV() << unBit++);
      unStatus |= (GEBStatus->bxmAvV() << unBit++);
      unStatus |= (GEBStatus->oOScVvV() << unBit++);
      unStatus |= (GEBStatus->oOScAvV() << unBit++);
      unStatus |= (GEBStatus->noVFAT() << unBit++);
      unStatus |= (GEBStatus->evtSzW() << unBit++);
      unStatus |= (GEBStatus->l1aNF() << unBit++);
      unStatus |= (GEBStatus->inNF() << unBit++);
      unStatus |= (GEBStatus->evtNF() << unBit++);
      unStatus |= (GEBStatus->evtSzOFW() << unBit++);
      unStatus |= (GEBStatus->l1aF() << unBit++);
      unStatus |= (GEBStatus->inF() << unBit++);
      unStatus |= (GEBStatus->evtF() << unBit++);
      unStatus |= (GEBStatus->inUfw() << unBit++);
      unStatus |= (GEBStatus->stuckData() << unBit++);
      unStatus |= (GEBStatus->evUfw() << unBit++);

      //std::cout << gemid;
      //printf("%06X\n", (unsigned int)unStatus);
      if (unStatus != 0) {  // Error!
        m_mapStatusErr[gemid] = true;
      }

      FillBits(listGEBInputStatus_[lid], unStatus, eBit_, nCh);

      listGEBInputID_[lid]->Fill(nCh, GEBStatus->inputID());
      listGEBVFATWordCnt_[lid]->Fill(nCh, GEBStatus->vfatWordCnt() / 3);
      listGEBVFATWordCntT_[lid]->Fill(nCh, GEBStatus->vfatWordCntT() / 3);
      listGEBZeroSupWordsCnt_[lid]->Fill(nCh, GEBStatus->zeroSupWordsCnt());

      listGEBbcOH_[lid]->Fill(nCh, GEBStatus->bcOH());
      listGEBecOH_[lid]->Fill(nCh, GEBStatus->ecOH());
      listGEBOHCRC_[lid]->Fill(nCh, GEBStatus->crc());

      fillTimeHisto(listCurr, nCh, unStatus != 0);
    }
  }

  auto findAMCIdx = [this](Int_t nAMCnum) -> Int_t {
    for (Int_t i = 0; i < (Int_t)listAMCSlots_.size(); i++)
      if (listAMCSlots_[i] == nAMCnum)
        return i;
    return -1;
  };

  for (GEMAMCdataCollection::DigiRangeIterator amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt) {
    const GEMAMCdataCollection::Range &range = (*amcIt).second;
    auto &listCurr = listTimeStore_[0];
    for (auto amc = range.first; amc != range.second; ++amc) {
      Int_t nIdAMC = findAMCIdx(amc->amcNum());
      uint64_t unBit = 0;
      uint64_t unStatus = 0;

      unStatus |= ((amc->bc0locked() == 0 ? 1 : 0) << unBit++);
      unStatus |= ((amc->daqReady() != 1 ? 1 : 0) << unBit++);
      unStatus |= ((amc->daqClockLocked() == 0 ? 1 : 0) << unBit++);
      unStatus |= ((amc->mmcmLocked() != 1 ? 1 : 0) << unBit++);
      unStatus |= ((amc->backPressure() == 1 ? 1 : 0) << unBit++);
      unStatus |= ((amc->oosGlib() != 0 ? 1 : 0) << unBit++);

      FillBits(h2AMCStatus_, unStatus, amcStatusBit_, nIdAMC);

      h1_amc_ttsState_->Fill(amc->ttsState());
      h1_amc_davCnt_->Fill(amc->davCnt());
      h1_amc_buffState_->Fill(amc->buffState());
      h1_amc_oosGlib_->Fill(amc->oosGlib());
      h1_amc_chTimeOut_->Fill(amc->linkTo());

      fillTimeHisto(listCurr, nIdAMC, unStatus != 0);
    }
  }

  /*auto findVFAT = [this](float min_, int nNumStripPerVFAT, float x_, int roll_)->int {
    float step = max_/3;
    if ( x_ < (min_+step) ) { return 8 - roll_;}
    else if ( x_ < (min_+2*step) )  { return 16 - roll_;}
    else { return this->nVfat_ - roll_;}
  };*/

  // Checking if there is a fire (data)
  for (auto ch : gemChambers_) {
    GEMDetId cId = ch.id();
    Bool_t bIsHit = false;

    // Because every fired strip in a same VFAT shares a same bx, we keep bx from only one strip
    std::unordered_map<Int_t, Int_t> mapBXVFAT;

    GEMDetId chIdDigi(cId.region(), cId.ring(), cId.station(), cId.layer(), cId.chamber(), 2);
    GEMDetId chIdBx(cId.region(), cId.ring(), cId.station(), cId.layer(), cId.chamber(), 3);

    auto &listCurrDigi = listTimeStore_[chIdDigi];
    auto &listCurrBx = listTimeStore_[chIdBx];

    for (auto roll : ch.etaPartitions()) {
      GEMDetId rId = roll->id();
      const auto &digis_in_det = gemDigis->get(rId);

      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        //Int_t nVFAT = findVFAT(0, roll->nstrips(), d->strip(), rId.roll()); // Starts at 0
        Int_t nIdxStrip = d->strip() - nIdxFirstStrip_;
        Int_t nVFAT = 8 * ((Int_t)(nIdxStrip / (roll->nstrips() / 3)) + 1) - rId.roll();  // Strip:Start at 0

        bIsHit = true;
        mapBXVFAT[nVFAT] = d->bx();
        fillTimeHisto(listCurrDigi, nVFAT, true);

        Int_t nIdxBx;

        if (d->bx() < listCurrBx.nNbinMin)
          nIdxBx = 0;
        else if (d->bx() >= listCurrBx.nNbinMax)
          nIdxBx = listCurrBx.nNbinY - 1;
        else
          nIdxBx = (Int_t)(listCurrBx.nNbinY * 1.0 * (d->bx() - listCurrBx.nNbinMin) /
                           (listCurrBx.nNbinMax - listCurrBx.nNbinMin));

        fillTimeHisto(listCurrBx, nIdxBx, true);
      }

      if (bIsHit)
        break;
    }

    for (auto bx : mapBXVFAT)
      listVFATBC_[cId]->Fill(bx.second, bx.first);

    if (bIsHit) {  // Data occur!
      m_mapStatusFill[cId] = true;
    }
  }

  // Counting the time tables
  nStackedEvt_++;
  if (nStackedEvt_ >= nNSecPerBin_ * nNEvtPerSec_) {  // Time to jump!
    for (auto &itStore : listTimeStore_)
      for (Int_t i = 0; i < itStore.second.nNbinY; i++) {
        itStore.second.listOccupy[i].push_back(0);
        itStore.second.listRecord[i].push_back(0);
      }

    nStackedEvt_ = 0;
  }
}

void GEMDQMStatusDigi::endRun(edm::Run const &run, edm::EventSetup const &eSetup) {
  //std::cout << "End run" << std::endl;

  for (auto chamber : gemChambers_) {
    auto gid = chamber.id();

    UInt_t unVal = 0;  // No data, no error
    if (m_mapStatusErr[gid])
      unVal = 2;  // Error! (no matter there are data or not)
    else if (m_mapStatusFill[gid])
      unVal = 1;  // Data occur, with no error
    //std::cout << gid << ": " << unVal << std::endl;

    Int_t nLayer = (bPerSuperchamber_ ? gid.layer() : 0);
    GEMDetId layerId(gid.region(), gid.ring(), gid.station(), nLayer, 0, 0);
    GEMDetId chamberId(0, 1, 1, gid.layer() - nLayer, gid.chamber(), 0);

    Int_t nIdxLayer = seekIdx(m_listLayers, layerId);
    Int_t nIdxChamber = seekIdx(m_listChambers, chamberId);

    if (h2SummaryStatus_)
      h2SummaryStatus_->setBinContent(nIdxChamber, nIdxLayer + 1, unVal);

    // The below is for under/overflow of BX histograms
    Int_t nNbinX = listVFATBC_[gid]->getNbinsX();
    Int_t nNbinY = listVFATBC_[gid]->getNbinsY();

    for (Int_t i = 0; i < nNbinY; i++) {
      listVFATBC_[gid]->setBinContent(
          1, i, listVFATBC_[gid]->getBinContent(0, i) + listVFATBC_[gid]->getBinContent(1, i));
      listVFATBC_[gid]->setBinContent(
          nNbinX, i, listVFATBC_[gid]->getBinContent(nNbinX, i) + listVFATBC_[gid]->getBinContent(nNbinX + 1, i));
    }
  }

  for (auto &itStore : listTimeStore_) {
    auto &listCurrOcc = itStore.second.listOccupy;
    auto &listCurrRec = itStore.second.listRecord;
    MonitorElement *h2Curr = itStore.second.h2Histo;

    h2Curr->setBinContent(0, 0, nNSecPerBin_ * nNEvtPerSec_);
    h2Curr->setBinContent(0, 1, 1);

    Int_t nSize = listCurrRec[0].size();
    //Int_t nXOffset = std::max(nSize - nNTimeBinTotal_, 0);
    Int_t nXOffset = 0;
    Int_t nNbinY = listCurrRec.size();

    for (Int_t nX = nXOffset; nX < nSize; nX++) {
      Int_t nNEvt = (nX < nSize - 1 ? nNEvtPerSec_ * nNSecPerBin_ : nStackedEvt_);
      h2Curr->setBinContent(nX - nXOffset + 1, 0, nNEvt);

      for (Int_t nY = 1; nY <= nNbinY; nY++) {
        Double_t dVal = listCurrRec[nY - 1][nX];
        Double_t dErr = listCurrOcc[nY - 1][nX];

        if (dErr > 0 && dVal <= 0)
          dVal = -1;

        if (dVal != 0)
          h2Curr->setBinContent(nX - nXOffset + 1, nY, dVal);
        //if ( dErr > 0 ) h2Curr->setBinError(nX - nXOffset + 1, nY, 1);
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMStatusDigi);
