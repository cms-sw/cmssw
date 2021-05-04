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
#include "DataFormats/GEMDigi/interface/GEMVfatStatusDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMGEBdataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCdataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13EventCollection.h"

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

private:
  int ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) override;

  void SetLabelAMCStatus(MonitorElement *h2Status);
  void SetLabelGEBStatus(MonitorElement *h2Status);
  void SetLabelVFATStatus(MonitorElement *h2Status);

  edm::EDGetToken tagDigi_;
  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagGEB_;
  edm::EDGetToken tagAMC_;
  edm::EDGetToken tagAMC13_;

  MonitorElement *h2AMCStatusPos_;
  MonitorElement *h2AMCStatusNeg_;
  MonitorElement *h2AMCNumGEBPos_;
  MonitorElement *h2AMCNumGEBNeg_;

  MEMap3Inf mapStatusGEB_;
  MEMap3Inf mapStatusVFAT_;
  MEMap3Inf mapGEBNumVFAT_;

  MEMap4Inf mapStatusVFATPerCh_;

  MonitorElement *h2SummaryStatus;

  Int_t nBXMin_, nBXMax_;

  std::map<UInt_t, int> mapFEDIdToRe_;
  Int_t nAMCSlots_;

  int nBitAMC_ = 7;
  int nBitGEB_ = 17;
  int nBitVFAT_ = 9;
};

using namespace std;
using namespace edm;

GEMDAQStatusSource::GEMDAQStatusSource(const edm::ParameterSet &cfg) : GEMDQMBase(cfg) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
  tagVFAT_ = consumes<GEMVfatStatusDigiCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel"));
  tagGEB_ = consumes<GEMGEBdataCollection>(cfg.getParameter<edm::InputTag>("GEBInputLabel"));
  tagAMC_ = consumes<GEMAMCdataCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel"));
  tagAMC13_ = consumes<GEMAMC13EventCollection>(cfg.getParameter<edm::InputTag>("AMC13InputLabel"));

  nAMCSlots_ = cfg.getParameter<Int_t>("AMCSlots");
}

void GEMDAQStatusSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "vfatStatus"));
  desc.add<edm::InputTag>("GEBInputLabel", edm::InputTag("muonGEMDigis", "gebStatus"));
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCdata"));
  desc.add<edm::InputTag>("AMC13InputLabel", edm::InputTag("muonGEMDigis", "AMC13Event"));

  desc.add<Int_t>("AMCSlots", 13);
  desc.addUntracked<std::string>("logCategory", "GEMDAQStatusSource");

  descriptions.add("GEMDAQStatusSource", desc);
}

void GEMDAQStatusSource::SetLabelAMCStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "BC0 not locked", 2);
  h2Status->setBinLabel(unBinPos++, "DAQ not ready", 2);
  h2Status->setBinLabel(unBinPos++, "DAQ clock not locked", 2);
  h2Status->setBinLabel(unBinPos++, "MMCM not locked", 2);
  h2Status->setBinLabel(unBinPos++, "Back pressure", 2);
  h2Status->setBinLabel(unBinPos++, "GLIB out-of-sync", 2);
}

void GEMDAQStatusSource::SetLabelGEBStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "BX mismatch GLIB OH", 2);
  h2Status->setBinLabel(unBinPos++, "BX mismatch GLIB VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "OOS GLIB OH", 2);
  h2Status->setBinLabel(unBinPos++, "OOS GLIB VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "No VFAT marker", 2);
  h2Status->setBinLabel(unBinPos++, "Event size warn", 2);
  h2Status->setBinLabel(unBinPos++, "L1AFIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "InFIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "EvtFIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "Event size overflow", 2);
  h2Status->setBinLabel(unBinPos++, "L1AFIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "InFIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "EvtFIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO underflow", 2);
  h2Status->setBinLabel(unBinPos++, "Stuck data", 2);
  h2Status->setBinLabel(unBinPos++, "Event FIFO underflow", 2);
}

void GEMDAQStatusSource::SetLabelVFATStatus(MonitorElement *h2Status) {
  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "CRC fail", 2);
  h2Status->setBinLabel(unBinPos++, "b1010 fail", 2);
  h2Status->setBinLabel(unBinPos++, "b1100 fail", 2);
  h2Status->setBinLabel(unBinPos++, "b1110 fail", 2);
  h2Status->setBinLabel(unBinPos++, "Hamming error", 2);
  h2Status->setBinLabel(unBinPos++, "AFULL", 2);
  h2Status->setBinLabel(unBinPos++, "SEUlogic", 2);
  h2Status->setBinLabel(unBinPos++, "SUEI2C", 2);
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
  h2AMCNumGEBPos_ = ibooker.book2D("amc_numGEBsPos",
                                   "Number of GEBs in AMCs (positive region);AMC slot;Number of GEBs",
                                   nAMCSlots_,
                                   -0.5,
                                   nAMCSlots_ - 0.5,
                                   41,
                                   -0.5,
                                   41 - 0.5);
  h2AMCNumGEBNeg_ = ibooker.book2D("amc_numGEBsNeg",
                                   "Number of GEBs in AMCs (negative region);AMC slot;Number of GEBs",
                                   nAMCSlots_,
                                   -0.5,
                                   nAMCSlots_ - 0.5,
                                   41,
                                   -0.5,
                                   41 - 0.5);

  SetLabelAMCStatus(h2AMCStatusPos_);
  SetLabelAMCStatus(h2AMCStatusNeg_);

  mapStatusGEB_ =
      MEMap3Inf(this, "geb_input_status", "GEB Input Status", 36, 0.5, 36.5, nBitGEB_, 0.5, nBitGEB_ + 0.5, "Chamber");
  mapStatusVFAT_ =
      MEMap3Inf(this, "vfat_status", "VFAT Quality Status", 24, 0.5, 24.5, nBitVFAT_, 0.5, nBitVFAT_ + 0.5, "VFAT");
  mapGEBNumVFAT_ = MEMap3Inf(this,
                             "geb_numVFATs",
                             "Number of VFATs in GEBs",
                             36,
                             0.5,
                             36.5,
                             24 + 1,
                             -0.5,
                             24 + 0.5);  // FIXME: The maximum number of VFATs is different for each stations

  mapStatusVFATPerCh_ =
      MEMap4Inf(this, "vfat_status", "VFAT Quality Status", 24, 0.5, 24.5, nBitVFAT_, 0.5, nBitVFAT_ + 0.5, "VFAT");

  GenerateMEPerChamber(ibooker);

  h2SummaryStatus = CreateSummaryHist(ibooker, "summaryStatus");
}

int GEMDAQStatusSource::ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) {
  MEStationInfo &stationInfo = mapStationInfo_[key];

  mapStatusGEB_.SetBinConfX(stationInfo.nNumChambers_);
  mapStatusGEB_.bookND(bh, key);
  mapStatusGEB_.SetLabelForChambers(key, 1);

  SetLabelGEBStatus(mapStatusGEB_.FindHist(key));

  mapStatusVFAT_.SetBinConfX(stationInfo.nMaxVFAT_);
  mapStatusVFAT_.bookND(bh, key);
  mapStatusVFAT_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 1);

  SetLabelVFATStatus(mapStatusVFAT_.FindHist(key));

  mapGEBNumVFAT_.SetBinConfX(stationInfo.nNumChambers_);
  mapGEBNumVFAT_.bookND(bh, key);
  mapGEBNumVFAT_.SetLabelForChambers(key, 1);

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
  edm::Handle<GEMVfatStatusDigiCollection> gemVFAT;
  edm::Handle<GEMGEBdataCollection> gemGEB;
  edm::Handle<GEMAMCdataCollection> gemAMC;
  edm::Handle<GEMAMC13EventCollection> gemAMC13;

  event.getByToken(tagDigi_, gemDigis);
  event.getByToken(tagVFAT_, gemVFAT);
  event.getByToken(tagGEB_, gemGEB);
  event.getByToken(tagAMC_, gemAMC);
  event.getByToken(tagAMC13_, gemAMC13);

  std::vector<int> listAMCRegion;

  for (GEMAMC13EventCollection::DigiRangeIterator amc13It = gemAMC13->begin(); amc13It != gemAMC13->end(); ++amc13It) {
    const GEMAMC13EventCollection::Range &range = (*amc13It).second;
    for (auto amc13 = range.first; amc13 != range.second; ++amc13) {
      for (int r = 0; r < int(amc13->nAMC()); r++) {
        listAMCRegion.push_back(mapFEDIdToRe_[(UInt_t)amc13->sourceId()]);
      }
    }
  }

  int nIdxAMCFull = 0;
  MonitorElement *h2AMCStatus, *h2AMCNumGEB;

  for (GEMAMCdataCollection::DigiRangeIterator amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt) {
    const GEMAMCdataCollection::Range &range = (*amcIt).second;
    for (auto amc = range.first; amc != range.second; ++amc) {
      uint64_t unBit = 1;
      bool bErr = false;

      if (nIdxAMCFull >= (int)listAMCRegion.size()) {
        edm::LogError(log_category_) << "+++ Error : Mismatch of the number of AMCs in AMC13 and the actual AMCs +++\n";
        break;
      }

      if (listAMCRegion[nIdxAMCFull] > 0) {
        h2AMCStatus = h2AMCStatusPos_;
        h2AMCNumGEB = h2AMCNumGEBPos_;
      } else {
        h2AMCStatus = h2AMCStatusNeg_;
        h2AMCNumGEB = h2AMCNumGEBNeg_;
      }

      unBit++;
      if (!amc->bc0locked()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      unBit++;
      if (!amc->daqReady()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      unBit++;
      if (!amc->daqClockLocked()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      unBit++;
      if (!amc->mmcmLocked()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      unBit++;
      if (amc->backPressure()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      unBit++;
      if (amc->oosGlib()) {
        h2AMCStatus->Fill(amc->amcNum(), unBit);
        bErr = true;
      }
      if (!bErr)
        h2AMCStatus->Fill(amc->amcNum(), 1);

      h2AMCNumGEB->Fill(amc->amcNum(), amc->gebs()->size());

      nIdxAMCFull++;
    }
  }

  // WARNING: ME4IdsKey for region, station, layer, chamber (not iEta)
  std::map<ME4IdsKey, bool> mapChamberStatus;

  for (GEMGEBdataCollection::DigiRangeIterator gebIt = gemGEB->begin(); gebIt != gemGEB->end(); ++gebIt) {
    GEMDetId gid = (*gebIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta

    const GEMGEBdataCollection::Range &range = (*gebIt).second;
    for (auto GEBStatus = range.first; GEBStatus != range.second; ++GEBStatus) {
      uint64_t unBit = 1;
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

      if (unStatus == 0) {
        unStatus = 0x01;  // Good
      } else {            // Error!
        mapChamberStatus[key4] = false;
      }

      mapStatusGEB_.FillBits(key3, gid.chamber(), unStatus);

      mapGEBNumVFAT_.Fill(key3, gid.chamber(), GEBStatus->vFATs()->size());
    }
  }

  for (GEMVfatStatusDigiCollection::DigiRangeIterator vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt) {
    GEMDetId gid = (*vfatIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta
    const GEMVfatStatusDigiCollection::Range &range = (*vfatIt).second;

    for (auto vfatStat = range.first; vfatStat != range.second; ++vfatStat) {
      uint64_t unQFVFAT = vfatStat->quality();
      if ((unQFVFAT & ~0x1) == 0) {
        unQFVFAT |= 0x1;  // If no error, then it should be 'Good'
      } else {            // Error!
        mapChamberStatus[key4Ch] = false;
      }

      Int_t nIdxVFAT = getVFATNumber(gid.station(), gid.roll(), vfatStat->phi());
      mapStatusVFAT_.FillBits(key3, nIdxVFAT + 1, unQFVFAT);
      mapStatusVFATPerCh_.FillBits(key4Ch, nIdxVFAT + 1, unQFVFAT);
    }
  }

  // Summarizing the error occupancy
  for (auto const &[key4, bErr] : mapChamberStatus) {
    ME3IdsKey key3 = key4Tokey3(key4);
    Int_t nChamber = keyToChamber(key4);
    h2SummaryStatus->Fill(nChamber, mapStationToIdx_[key3]);
  }
}

DEFINE_FWK_MODULE(GEMDAQStatusSource);
