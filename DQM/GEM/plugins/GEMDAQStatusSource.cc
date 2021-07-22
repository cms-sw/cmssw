#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13StatusCollection.h"

#include "DQM/GEM/interface/GEMDQMBase.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDAQStatusSource : public GEMDQMBase {
public:
  explicit GEMDAQStatusSource(const edm::ParameterSet &cfg);
  ~GEMDAQStatusSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override{};
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

  void FillWithRiseErr(MonitorElement *h, Int_t nX, Int_t nY, Bool_t &bErr) {
    h->Fill(nX, nY);
    bErr = true;
  };

private:
  int ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) override;

  void SetLabelAMC13Status(MonitorElement *h2Status);
  void SetLabelAMCStatus(MonitorElement *h2Status);
  void SetLabelOHStatus(MonitorElement *h2Status);
  void SetLabelVFATStatus(MonitorElement *h2Status);

  edm::EDGetToken tagDigi_;
  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagOH_;
  edm::EDGetToken tagAMC_;
  edm::EDGetToken tagAMC13_;

  MonitorElement *h2AMC13Status_;
  MonitorElement *h2AMCStatusPos_;
  MonitorElement *h2AMCStatusNeg_;
  MonitorElement *h2AMCNumOHPos_;
  MonitorElement *h2AMCNumOHNeg_;

  MEMap3Inf mapStatusOH_;
  MEMap3Inf mapStatusVFAT_;
  MEMap3Inf mapOHNumVFAT_;

  MEMap3Inf mapStatusWarnVFATPerLayer_;
  MEMap3Inf mapStatusErrVFATPerLayer_;
  MEMap4Inf mapStatusVFATPerCh_;

  MonitorElement *h2SummaryStatusWarning;
  MonitorElement *h2SummaryStatusError;

  Int_t nBXMin_, nBXMax_;

  std::map<UInt_t, int> mapFEDIdToRe_;
  Int_t nAMCSlots_;

  int nBitAMC13_ = 10;
  int nBitAMC_ = 13;
  int nBitOH_ = 17;
  int nBitVFAT_ = 7;
};

using namespace std;
using namespace edm;

GEMDAQStatusSource::GEMDAQStatusSource(const edm::ParameterSet &cfg) : GEMDQMBase(cfg) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
  tagVFAT_ = consumes<GEMVFATStatusCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel"));
  tagOH_ = consumes<GEMOHStatusCollection>(cfg.getParameter<edm::InputTag>("OHInputLabel"));
  tagAMC_ = consumes<GEMAMCStatusCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel"));
  tagAMC13_ = consumes<GEMAMC13StatusCollection>(cfg.getParameter<edm::InputTag>("AMC13InputLabel"));

  nAMCSlots_ = cfg.getParameter<Int_t>("AMCSlots");
}

void GEMDAQStatusSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "VFATStatus"));
  desc.add<edm::InputTag>("OHInputLabel", edm::InputTag("muonGEMDigis", "OHStatus"));
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCStatus"));
  desc.add<edm::InputTag>("AMC13InputLabel", edm::InputTag("muonGEMDigis", "AMC13Status"));

  desc.add<Int_t>("AMCSlots", 13);
  desc.addUntracked<std::string>("logCategory", "GEMDAQStatusSource");

  descriptions.add("GEMDAQStatusSource", desc);
}

void GEMDAQStatusSource::SetLabelAMC13Status(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid AMC", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid size", 2);
  h2Status->setBinLabel(unBinPos++, "Fail on trailer check", 2);
  h2Status->setBinLabel(unBinPos++, "Fail on fragment length", 2);
  h2Status->setBinLabel(unBinPos++, "Fail on trailer match", 2);
  h2Status->setBinLabel(unBinPos++, "More trailer", 2);
  h2Status->setBinLabel(unBinPos++, "CRC modified", 2);
  h2Status->setBinLabel(unBinPos++, "S-link error", 2);
  h2Status->setBinLabel(unBinPos++, "Wrong FED ID", 2);
}

void GEMDAQStatusSource::SetLabelAMCStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid OH", 2);
  h2Status->setBinLabel(unBinPos++, "Back pressure", 2);
  h2Status->setBinLabel(unBinPos++, "Bad EC", 2);
  h2Status->setBinLabel(unBinPos++, "Bad BC", 2);
  h2Status->setBinLabel(unBinPos++, "Bad OC", 2);
  h2Status->setBinLabel(unBinPos++, "Bad run type", 2);
  h2Status->setBinLabel(unBinPos++, "Bad CRC", 2);
  h2Status->setBinLabel(unBinPos++, "MMCM locked", 2);
  h2Status->setBinLabel(unBinPos++, "DAQ clock locked", 2);
  h2Status->setBinLabel(unBinPos++, "DAQ not ready", 2);
  h2Status->setBinLabel(unBinPos++, "BC0 not locked", 2);
}

void GEMDAQStatusSource::SetLabelOHStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Event FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "L1A FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "Event size warn", 2);
  h2Status->setBinLabel(unBinPos++, "Event FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "L1A FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "Event size overflow", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid event", 2);
  h2Status->setBinLabel(unBinPos++, "Out of Sync AMC vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "Out of Sync VFAT vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "BX mismatch AMC vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "1st bit BX mismatch VFAT vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO underflow", 2);
}

void GEMDAQStatusSource::SetLabelVFATStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Basic overflow", 2);
  h2Status->setBinLabel(unBinPos++, "Zero-sup overflow", 2);
  h2Status->setBinLabel(unBinPos++, "VFAT CRC error", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid header", 2);
  h2Status->setBinLabel(unBinPos++, "No match AMC EC", 2);
  h2Status->setBinLabel(unBinPos++, "No match AMC BC", 2);
}

void GEMDAQStatusSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  nBXMin_ = -10;
  nBXMax_ = 10;

  mapFEDIdToRe_[1467] = -1;  // FIXME: Need more systematic way
  mapFEDIdToRe_[1468] = 1;

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/DAQStatus");

  h2AMC13Status_ = ibooker.book2D("amc13_statusflag",
                                  "Status of AMC13;AMC13;",
                                  2,
                                  -0.5,
                                  0.5,
                                  nBitAMC13_,
                                  0.5,
                                  nBitAMC13_ + 0.5);
  h2AMCStatusPos_ = ibooker.book2D("amc_statusflagPos",
                                   "Status of AMC slots (positive region);AMC slot;",
                                   nAMCSlots_,
                                   -0.5,
                                   nAMCSlots_ - 0.5,
                                   nBitAMC_,
                                   0.5,
                                   nBitAMC_ + 0.5);
  h2AMCStatusNeg_ = ibooker.book2D("amc_statusflagNeg",
                                   "Status of AMC slots (negative region);AMC slot;",
                                   nAMCSlots_,
                                   -0.5,
                                   nAMCSlots_ - 0.5,
                                   nBitAMC_,
                                   0.5,
                                   nBitAMC_ + 0.5);
  h2AMCNumOHPos_ = ibooker.book2D("amc_numOHsPos",
                                  "Number of OHs in AMCs (positive region);AMC slot;Number of OHs",
                                  nAMCSlots_,
                                  -0.5,
                                  nAMCSlots_ - 0.5,
                                  41,
                                  -0.5,
                                  41 - 0.5);
  h2AMCNumOHNeg_ = ibooker.book2D("amc_numOHsNeg",
                                  "Number of OHs in AMCs (negative region);AMC slot;Number of OHs",
                                  nAMCSlots_,
                                  -0.5,
                                  nAMCSlots_ - 0.5,
                                  41,
                                  -0.5,
                                  41 - 0.5);

  SetLabelAMC13Status(h2AMC13Status_);
  SetLabelAMCStatus(h2AMCStatusPos_);
  SetLabelAMCStatus(h2AMCStatusNeg_);

  mapStatusOH_ =
      MEMap3Inf(this, "oh_input_status", "OctoHybrid Input Status", 36, 0.5, 36.5, nBitOH_, 0.5, nBitOH_ + 0.5, "Chamber");
  mapStatusVFAT_ =
      MEMap3Inf(this, "vfat_status", "VFAT Quality Status", 24, 0.5, 24.5, nBitVFAT_, 0.5, nBitVFAT_ + 0.5, "VFAT");
  mapOHNumVFAT_ = MEMap3Inf(this,
                             "oh_numVFATs",
                             "Number of VFATs in OHs",
                             36,
                             0.5,
                             36.5,
                             24 + 1,
                             -0.5,
                             24 + 0.5);  // FIXME: The maximum number of VFATs is different for each stations

  mapStatusWarnVFATPerLayer_ = MEMap3Inf(
      this, "vfat_statusWarnSum", "Summary on VFAT Quality Status", 36, 0.5, 36.5, 24, 0.5, 24.5, "Chamber", "VFAT");
  mapStatusErrVFATPerLayer_ = MEMap3Inf(
      this, "vfat_statusErrSum", "Summary on VFAT Quality Status", 36, 0.5, 36.5, 24, 0.5, 24.5, "Chamber", "VFAT");
  mapStatusVFATPerCh_ =
      MEMap4Inf(this, "vfat_status", "VFAT Quality Status", 24, 0.5, 24.5, nBitVFAT_, 0.5, nBitVFAT_ + 0.5, "VFAT");

  GenerateMEPerChamber(ibooker);

  h2SummaryStatusWarning = CreateSummaryHist(ibooker, "summaryStatusWarning");
  h2SummaryStatusError   = CreateSummaryHist(ibooker, "summaryStatusError");
}

int GEMDAQStatusSource::ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) {
  MEStationInfo &stationInfo = mapStationInfo_[key];

  mapStatusOH_.SetBinConfX(stationInfo.nNumChambers_);
  mapStatusOH_.bookND(bh, key);
  mapStatusOH_.SetLabelForChambers(key, 1);

  SetLabelOHStatus(mapStatusOH_.FindHist(key));

  mapStatusWarnVFATPerLayer_.SetBinConfX(stationInfo.nNumChambers_);
  mapStatusWarnVFATPerLayer_.SetBinConfY(stationInfo.nMaxVFAT_);
  mapStatusWarnVFATPerLayer_.bookND(bh, key);
  mapStatusWarnVFATPerLayer_.SetLabelForChambers(key, 1);
  mapStatusWarnVFATPerLayer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapStatusErrVFATPerLayer_.SetBinConfX(stationInfo.nNumChambers_);
  mapStatusErrVFATPerLayer_.SetBinConfY(stationInfo.nMaxVFAT_);
  mapStatusErrVFATPerLayer_.bookND(bh, key);
  mapStatusErrVFATPerLayer_.SetLabelForChambers(key, 1);
  mapStatusErrVFATPerLayer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapStatusVFAT_.SetBinConfX(stationInfo.nMaxVFAT_);
  mapStatusVFAT_.bookND(bh, key);
  mapStatusVFAT_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 1);

  SetLabelVFATStatus(mapStatusVFAT_.FindHist(key));

  mapOHNumVFAT_.SetBinConfX(stationInfo.nNumChambers_);
  mapOHNumVFAT_.bookND(bh, key);
  mapOHNumVFAT_.SetLabelForChambers(key, 1);

  return 0;
}

int GEMDAQStatusSource::ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo &stationInfo = mapStationInfo_[key3];

  mapStatusVFATPerCh_.SetBinConfX(stationInfo.nMaxVFAT_);
  mapStatusVFATPerCh_.bookND(bh, key);
  mapStatusVFATPerCh_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 1);
  SetLabelVFATStatus(mapStatusVFATPerCh_.FindHist(key));

  return 0;
}

void GEMDAQStatusSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  edm::Handle<GEMDigiCollection> gemDigis;
  edm::Handle<GEMVFATStatusCollection> gemVFAT;
  edm::Handle<GEMOHStatusCollection> gemOH;
  edm::Handle<GEMAMCStatusCollection> gemAMC;
  edm::Handle<GEMAMC13StatusCollection> gemAMC13;

  event.getByToken(tagDigi_, gemDigis);
  event.getByToken(tagVFAT_, gemVFAT);
  event.getByToken(tagOH_, gemOH);
  event.getByToken(tagAMC_, gemAMC);
  event.getByToken(tagAMC13_, gemAMC13);
  
  for (auto amc13It = gemAMC13->begin(); amc13It != gemAMC13->end(); ++amc13It) {
    int fedId = (*amc13It).first;
    if (mapFEDIdToRe_.find(fedId) == mapFEDIdToRe_.end()) continue;
    int nXBin = 0;
    if (mapFEDIdToRe_[fedId] > 0 ) {
      nXBin = 1;
    } else if (mapFEDIdToRe_[fedId] < 0) {
      nXBin = 2;
    } else {
      edm::LogError(log_category_) << "+++ Error : Unknown FED Id +++\n" << std::endl;
      continue;
    }

    const auto &range = (*amc13It).second;
    for (auto amc13 = range.first; amc13 != range.second; ++amc13) {
      Bool_t bWarn = false;
      Bool_t bErr  = false;

      GEMAMC13Status::Warnings warnings{amc13->warnings()};
      if (warnings.InValidAMC) FillWithRiseErr(h2AMC13Status_, nXBin, 2, bWarn);

      GEMAMC13Status::Errors errors{amc13->errors()};
      if (errors.InValidSize)        FillWithRiseErr(h2AMC13Status_, nXBin,  3, bErr);
      if (errors.failTrailerCheck)   FillWithRiseErr(h2AMC13Status_, nXBin,  4, bErr);
      if (errors.failFragmentLength) FillWithRiseErr(h2AMC13Status_, nXBin,  5, bErr);
      if (errors.failTrailerMatch)   FillWithRiseErr(h2AMC13Status_, nXBin,  6, bErr);
      if (errors.moreTrailers)       FillWithRiseErr(h2AMC13Status_, nXBin,  7, bErr);
      if (errors.crcModified)        FillWithRiseErr(h2AMC13Status_, nXBin,  8, bErr);
      if (errors.slinkError)         FillWithRiseErr(h2AMC13Status_, nXBin,  9, bErr);
      if (errors.wrongFedId)         FillWithRiseErr(h2AMC13Status_, nXBin, 10, bErr);

      if (!bWarn && !bErr) h2AMC13Status_->Fill(nXBin, 1);
    }
  }

  MonitorElement *h2AMCStatus = nullptr;

  for (auto amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt) {
    int fedId = (*amcIt).first;
    if (mapFEDIdToRe_.find(fedId) == mapFEDIdToRe_.end()) continue;
    if (mapFEDIdToRe_[fedId] > 0) {
      h2AMCStatus = h2AMCStatusPos_;
    } else if (mapFEDIdToRe_[fedId] < 0) {
      h2AMCStatus = h2AMCStatusNeg_;
    } else {
      edm::LogError(log_category_) << "+++ Error : Unknown FED Id +++\n" << std::endl;
      continue;
    }

    const GEMAMCStatusCollection::Range &range = (*amcIt).second;
    for (auto amc = range.first; amc != range.second; ++amc) {
      Bool_t bWarn = false;
      Bool_t bErr  = false;

      Int_t nAMCNum = amc->amcNumber();

      GEMAMCStatus::Warnings warnings{amc->warnings()};
      if (warnings.InValidOH)    FillWithRiseErr(h2AMCStatus, nAMCNum, 2, bWarn);
      if (warnings.backPressure) FillWithRiseErr(h2AMCStatus, nAMCNum, 3, bWarn);

      GEMAMCStatus::Errors errors{amc->errors()};
      if (errors.badEC)          FillWithRiseErr(h2AMCStatus, nAMCNum,  4, bErr);
      if (errors.badBC)          FillWithRiseErr(h2AMCStatus, nAMCNum,  5, bErr);
      if (errors.badOC)          FillWithRiseErr(h2AMCStatus, nAMCNum,  6, bErr);
      if (errors.badRunType)     FillWithRiseErr(h2AMCStatus, nAMCNum,  7, bErr);
      if (errors.badCRC)         FillWithRiseErr(h2AMCStatus, nAMCNum,  8, bErr);
      if (errors.MMCMlocked)     FillWithRiseErr(h2AMCStatus, nAMCNum,  9, bErr);
      if (errors.DAQclocklocked) FillWithRiseErr(h2AMCStatus, nAMCNum, 10, bErr);
      if (errors.DAQnotReday)    FillWithRiseErr(h2AMCStatus, nAMCNum, 11, bErr);
      if (errors.BC0locked)      FillWithRiseErr(h2AMCStatus, nAMCNum, 12, bErr);

      if (!bWarn && !bErr) h2AMCStatus->Fill(nAMCNum, 1);
    }
  }

  // WARNING: ME4IdsKey for region, station, layer, chamber (not iEta)
  std::map<ME4IdsKey, bool> mapChamberWarning;
  std::map<ME4IdsKey, bool> mapChamberError;

  for (auto ohIt = gemOH->begin(); ohIt != gemOH->end(); ++ohIt) {
    GEMDetId gid = (*ohIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta

    const GEMOHStatusCollection::Range &range = (*ohIt).second;
    for (auto OHStatus = range.first; OHStatus != range.second; ++OHStatus) {
      Bool_t bWarn = false;
      Bool_t bErr  = false;

      GEMOHStatus::Warnings warnings{OHStatus->warnings()};
      if (warnings.EvtNF)       bWarn = mapStatusOH_.Fill(key3, gid.chamber(), 2) > 0;
      if (warnings.InNF)        bWarn = mapStatusOH_.Fill(key3, gid.chamber(), 3) > 0;
      if (warnings.L1aNF)       bWarn = mapStatusOH_.Fill(key3, gid.chamber(), 4) > 0;
      if (warnings.EvtSzW)      bWarn = mapStatusOH_.Fill(key3, gid.chamber(), 5) > 0;
      if (warnings.InValidVFAT) bWarn = mapStatusOH_.Fill(key3, gid.chamber(), 6) > 0;

      GEMOHStatus::Errors errors{OHStatus->errors()};
      if (errors.EvtF)         bErr = mapStatusOH_.Fill(key3, gid.chamber(),  7) > 0;
      if (errors.InF)          bErr = mapStatusOH_.Fill(key3, gid.chamber(),  8) > 0;
      if (errors.L1aF)         bErr = mapStatusOH_.Fill(key3, gid.chamber(),  9) > 0;
      if (errors.EvtSzOFW)     bErr = mapStatusOH_.Fill(key3, gid.chamber(), 10) > 0;
      if (errors.Inv)          bErr = mapStatusOH_.Fill(key3, gid.chamber(), 11) > 0;
      if (errors.OOScAvV)      bErr = mapStatusOH_.Fill(key3, gid.chamber(), 12) > 0;
      if (errors.OOScVvV)      bErr = mapStatusOH_.Fill(key3, gid.chamber(), 13) > 0;
      if (errors.BxmAvV)       bErr = mapStatusOH_.Fill(key3, gid.chamber(), 14) > 0;
      if (errors.BxmVvV)       bErr = mapStatusOH_.Fill(key3, gid.chamber(), 15) > 0;
      if (errors.InUfw)        bErr = mapStatusOH_.Fill(key3, gid.chamber(), 16) > 0;
      if (errors.badVFatCount) bErr = mapStatusOH_.Fill(key3, gid.chamber(), 17) > 0;

      if (!bWarn && !bErr) mapStatusOH_.Fill(key3, gid.chamber(), 1);
      if (bWarn) mapChamberWarning[key4] = false;
      if (bErr)  mapChamberError[key4] = false;
    }
  }

  for (auto vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt) {
    GEMDetId gid = (*vfatIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta
    const GEMVFATStatusCollection::Range &range = (*vfatIt).second;

    for (auto vfatStat = range.first; vfatStat != range.second; ++vfatStat) {
      // NOTE: nIdxVFAT starts from 1
      Int_t nIdxVFAT = getVFATNumber(gid.station(), gid.ieta(), vfatStat->vfatPosition()) + 1;
      Bool_t bWarn = false;
      Bool_t bErr  = false;

      GEMVFATStatus::Warnings warnings{vfatStat->warnings()};
      if (warnings.basicOFW)   bWarn = mapStatusVFAT_.Fill(key3, nIdxVFAT, 2);
      if (warnings.zeroSupOFW) bWarn = mapStatusVFAT_.Fill(key3, nIdxVFAT, 3);
      if (warnings.basicOFW)   bWarn = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 2);
      if (warnings.zeroSupOFW) bWarn = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 3);

      GEMVFATStatus::Errors errors{(uint8_t)vfatStat->errors()};
      if (errors.vc)            bErr = mapStatusVFAT_.Fill(key3, nIdxVFAT, 4);
      if (errors.InValidHeader) bErr = mapStatusVFAT_.Fill(key3, nIdxVFAT, 5);
      if (errors.EC)            bErr = mapStatusVFAT_.Fill(key3, nIdxVFAT, 6);
      if (errors.BC)            bErr = mapStatusVFAT_.Fill(key3, nIdxVFAT, 7);
      if (errors.vc)            bErr = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 4);
      if (errors.InValidHeader) bErr = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 5);
      if (errors.EC)            bErr = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 6);
      if (errors.BC)            bErr = mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 7);

      if (!bWarn && !bErr) mapStatusVFAT_.Fill(key3, nIdxVFAT, 1);
      if (!bWarn && !bErr) mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 1);
      if (bWarn) mapChamberWarning[key4Ch] = false;
      if (bErr)  mapChamberError[key4Ch] = false;
      if (bWarn) mapStatusWarnVFATPerLayer_.Fill(key3, gid.chamber(), nIdxVFAT);
      if (bErr)  mapStatusErrVFATPerLayer_.Fill(key3, gid.chamber(), nIdxVFAT);
    }
  }

  // Summarizing the warning occupancy
  for (auto const &[key4, bWarning] : mapChamberWarning) {
    ME3IdsKey key3 = key4Tokey3(key4);
    Int_t nChamber = keyToChamber(key4);
    h2SummaryStatusWarning->Fill(nChamber, mapStationToIdx_[key3]);
  }

  // Summarizing the error occupancy
  for (auto const &[key4, bErr] : mapChamberError) {
    ME3IdsKey key3 = key4Tokey3(key4);
    Int_t nChamber = keyToChamber(key4);
    h2SummaryStatusError->Fill(nChamber, mapStationToIdx_[key3]);
  }
}

DEFINE_FWK_MODULE(GEMDAQStatusSource);
