#include "DQM/GEM/interface/GEMDAQStatusSource.h"

using namespace std;
using namespace edm;

GEMDAQStatusSource::GEMDAQStatusSource(const edm::ParameterSet &cfg)
    : GEMDQMBase(cfg), gemChMapToken_(esConsumes<GEMChMap, GEMChMapRcd, edm::Transition::BeginRun>()) {
  tagVFAT_ = consumes<GEMVFATStatusCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel"));
  tagOH_ = consumes<GEMOHStatusCollection>(cfg.getParameter<edm::InputTag>("OHInputLabel"));
  tagAMC_ = consumes<GEMAMCStatusCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel"));
  tagAMC13_ = consumes<GEMAMC13StatusCollection>(cfg.getParameter<edm::InputTag>("AMC13InputLabel"));

  nAMCSlots_ = cfg.getParameter<Int_t>("AMCSlots");

  bWarnedNotFound_ = false;
}

void GEMDAQStatusSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "VFATStatus"));
  desc.add<edm::InputTag>("OHInputLabel", edm::InputTag("muonGEMDigis", "OHStatus"));
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCStatus"));
  desc.add<edm::InputTag>("AMC13InputLabel", edm::InputTag("muonGEMDigis", "AMC13Status"));

  desc.add<Int_t>("AMCSlots", 13);
  desc.addUntracked<std::string>("runType", "relval");
  desc.addUntracked<std::string>("logCategory", "GEMDAQStatusSource");

  descriptions.add("GEMDAQStatusSource", desc);
}

void GEMDAQStatusSource::LoadROMap(edm::EventSetup const &iSetup) {
  //if (useDBEMap_)
  if (true) {
    const auto &chMap = iSetup.getData(gemChMapToken_);
    auto gemChMap = std::make_unique<GEMChMap>(chMap);

    std::vector<unsigned int> listFEDId;
    for (auto const &[ec, dc] : gemChMap->chamberMap()) {
      unsigned int fedId = ec.fedId;
      uint8_t amcNum = ec.amcNum;
      GEMDetId gemChId(dc.detId);

      if (mapFEDIdToRe_.find(fedId) == mapFEDIdToRe_.end()) {
        listFEDId.push_back(fedId);
      }
      mapFEDIdToRe_[fedId] = gemChId.region();
      mapFEDIdToSt_[fedId] = gemChId.station();
      mapAMC13ToListChamber_[fedId].push_back(gemChId);
      mapAMCToListChamber_[{fedId, amcNum}].push_back(gemChId);
    }

    Int_t nIdx = 1;
    for (auto fedId : listFEDId) {
      mapFEDIdToPosition_[fedId] = nIdx++;
    }

  } else {
    // no EMap in DB, using dummy
    auto gemChMap = std::make_unique<GEMChMap>();
    gemChMap->setDummy();

    for (auto const &[ec, dc] : gemChMap->chamberMap()) {
      unsigned int fedId = ec.fedId;
      uint8_t amcNum = ec.amcNum;
      GEMDetId gemChId(dc.detId);

      mapFEDIdToRe_[fedId] = gemChId.region();
      mapAMC13ToListChamber_[fedId].push_back(gemChId);
      mapAMCToListChamber_[{fedId, amcNum}].push_back(gemChId);
    }
  }
}

void GEMDAQStatusSource::SetLabelAMC13Status(MonitorElement *h2Status) {
  if (h2Status == nullptr) {
    return;
  }

  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid AMC", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid size", 2);
  h2Status->setBinLabel(unBinPos++, "Fail trailer check", 2);
  h2Status->setBinLabel(unBinPos++, "Fail fragment length", 2);
  h2Status->setBinLabel(unBinPos++, "Fail trailer match", 2);
  h2Status->setBinLabel(unBinPos++, "More trailer", 2);
  h2Status->setBinLabel(unBinPos++, "CRC modified", 2);
  h2Status->setBinLabel(unBinPos++, "S-link error", 2);
  h2Status->setBinLabel(unBinPos++, "Wrong FED ID", 2);

  for (auto const &[fedId, nPos] : mapFEDIdToPosition_) {
    auto st = mapFEDIdToSt_[fedId];
    auto re = (mapFEDIdToRe_[fedId] > 0 ? 'P' : 'M');
    h2Status->setBinLabel(nPos, Form("GE%i1-%c", st, re), 1);
  }
}

void GEMDAQStatusSource::SetLabelAMCStatus(MonitorElement *h2Status) {
  if (h2Status == nullptr) {
    return;
  }

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
  if (h2Status == nullptr) {
    return;
  }

  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Event FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "L1A FIFO near full", 2);
  h2Status->setBinLabel(unBinPos++, "Event size warn", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "Event FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "L1A FIFO full", 2);
  h2Status->setBinLabel(unBinPos++, "Event size overflow", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid event", 2);
  h2Status->setBinLabel(unBinPos++, "Out of Sync AMC vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "Out of Sync VFAT vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "BX mismatch AMC vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "BX mismatch VFAT vs VFAT", 2);
  h2Status->setBinLabel(unBinPos++, "Input FIFO underflow", 2);
  h2Status->setBinLabel(unBinPos++, "Bad VFAT count", 2);
}

void GEMDAQStatusSource::SetLabelVFATStatus(MonitorElement *h2Status) {
  if (h2Status == nullptr) {
    return;
  }

  unsigned int unBinPos = 1;
  h2Status->setBinLabel(unBinPos++, "Good", 2);
  h2Status->setBinLabel(unBinPos++, "Basic overflow", 2);
  h2Status->setBinLabel(unBinPos++, "Zero-sup overflow", 2);
  h2Status->setBinLabel(unBinPos++, "VFAT CRC error", 2);
  h2Status->setBinLabel(unBinPos++, "Invalid header", 2);
  h2Status->setBinLabel(unBinPos++, "AMC EC mismatch", 2);
  h2Status->setBinLabel(unBinPos++, "AMC BC mismatch", 2);
}

void GEMDAQStatusSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &iSetup) {
  LoadROMap(iSetup);
  if (mapAMC13ToListChamber_.empty() || mapAMCToListChamber_.empty())
    return;
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  strFolderMain_ = "GEM/DAQStatus";

  nBXMin_ = -10;
  nBXMax_ = 10;

  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);

  h2AMC13Status_ = nullptr;

  bFillAMC_ = false;

  //if (nRunType_ != GEMDQM_RUNTYPE_RELVAL)
  if (nRunType_ == GEMDQM_RUNTYPE_ALLPLOTS || nRunType_ == GEMDQM_RUNTYPE_ONLINE) {
    Int_t nNumAMC13 = (Int_t)mapFEDIdToRe_.size();
    h2AMC13Status_ = ibooker.book2D(
        "amc13_status", "AMC13 Status;AMC13;", nNumAMC13, 0.5, nNumAMC13 + 0.5, nBitAMC13_, 0.5, nBitAMC13_ + 0.5);
    SetLabelAMC13Status(h2AMC13Status_);

    for (auto &[fedId, nIdx] : mapFEDIdToPosition_) {
      auto st = mapFEDIdToSt_[fedId];
      auto re = (mapFEDIdToRe_[fedId] > 0 ? 'P' : 'M');
      auto strName = Form("amc_status_GE%i1-%c", st, re);
      auto strTitle = Form("AMC Status GE%i1-%c;AMC slot;", st, re);
      mapFEDIdToAMCStatus_[fedId] =
          ibooker.book2D(strName, strTitle, nAMCSlots_, -0.5, nAMCSlots_ - 0.5, nBitAMC_, 0.5, nBitAMC_ + 0.5);
      SetLabelAMCStatus(mapFEDIdToAMCStatus_[fedId]);
    }

    bFillAMC_ = true;
  }

  mapStatusOH_ =
      MEMap3Inf(this, "oh_status", "OptoHybrid Status", 36, 0.5, 36.5, nBitOH_, 0.5, nBitOH_ + 0.5, "Chamber");

  mapStatusWarnVFATPerLayer_ = MEMap3Inf(
      this, "vfat_statusWarnSum", "VFAT reporting warnings", 36, 0.5, 36.5, 24, -0.5, 24 - 0.5, "Chamber", "VFAT");
  mapStatusErrVFATPerLayer_ = MEMap3Inf(
      this, "vfat_statusErrSum", "VFAT reporting errors", 36, 0.5, 36.5, 24, -0.5, 24 - 0.5, "Chamber", "VFAT");
  mapStatusVFATPerCh_ =
      MEMap4Inf(this, "vfat_status", "VFAT Status", 24, -0.5, 24 - 0.5, nBitVFAT_, 0.5, nBitVFAT_ + 0.5, "VFAT");

  if (nRunType_ == GEMDQM_RUNTYPE_OFFLINE || nRunType_ == GEMDQM_RUNTYPE_RELVAL) {
    mapStatusOH_.TurnOff();
    mapStatusWarnVFATPerLayer_.TurnOff();
    mapStatusErrVFATPerLayer_.TurnOff();
    mapStatusVFATPerCh_.TurnOff();
  }

  GenerateMEPerChamber(ibooker);

  if (nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
    h2SummaryStatusAll = CreateSummaryHist(ibooker, "chamberAllStatus");
    h2SummaryStatusWarning = CreateSummaryHist(ibooker, "chamberWarnings");
    h2SummaryStatusError = CreateSummaryHist(ibooker, "chamberErrors");
  }

  if (nRunType_ == GEMDQM_RUNTYPE_ALLPLOTS || nRunType_ == GEMDQM_RUNTYPE_ONLINE) {
    h2SummaryStatusVFATWarning = CreateSummaryHist(ibooker, "chamberVFATWarnings");
    h2SummaryStatusVFATError = CreateSummaryHist(ibooker, "chamberVFATErrors");
    h2SummaryStatusOHWarning = CreateSummaryHist(ibooker, "chamberOHWarnings");
    h2SummaryStatusOHError = CreateSummaryHist(ibooker, "chamberOHErrors");
    h2SummaryStatusAMCWarning = CreateSummaryHist(ibooker, "chamberAMCWarnings");
    h2SummaryStatusAMCError = CreateSummaryHist(ibooker, "chamberAMCErrors");
    h2SummaryStatusAMC13Error = CreateSummaryHist(ibooker, "chamberAMC13Errors");

    h2SummaryStatusAll->setTitle("Summary of all number of OH or VFAT status of each chambers");
    h2SummaryStatusWarning->setTitle("Summary of all warnings of each chambers");
    h2SummaryStatusError->setTitle("Summary of all errors of each chambers");
    h2SummaryStatusVFATWarning->setTitle("Summary of VFAT warnings of each chambers");
    h2SummaryStatusVFATError->setTitle("Summary of VFAT errors of each chambers");
    h2SummaryStatusOHWarning->setTitle("Summary of OH warnings of each chambers");
    h2SummaryStatusOHError->setTitle("Summary of OH errors of each chambers");
    h2SummaryStatusAMCWarning->setTitle("Summary of AMC warnings of each chambers");
    h2SummaryStatusAMCError->setTitle("Summary of AMC errors of each chambers");
    h2SummaryStatusAMC13Error->setTitle("Summary of AMC13 errors of each chambers");
  }
}

int GEMDAQStatusSource::ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) {
  MEStationInfo &stationInfo = mapStationInfo_[key];

  Int_t nNewNumCh = stationInfo.nMaxIdxChamber_ - stationInfo.nMinIdxChamber_ + 1;

  mapStatusOH_.SetBinConfX(nNewNumCh, stationInfo.nMinIdxChamber_ - 0.5, stationInfo.nMaxIdxChamber_ + 0.5);
  mapStatusOH_.bookND(bh, key);
  mapStatusOH_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);

  if (mapStatusOH_.isOperating()) {
    SetLabelOHStatus(mapStatusOH_.FindHist(key));
  }

  mapStatusWarnVFATPerLayer_.SetBinConfX(
      nNewNumCh, stationInfo.nMinIdxChamber_ - 0.5, stationInfo.nMaxIdxChamber_ + 0.5);
  mapStatusWarnVFATPerLayer_.SetBinConfY(stationInfo.nMaxVFAT_, -0.5);
  mapStatusWarnVFATPerLayer_.bookND(bh, key);
  mapStatusWarnVFATPerLayer_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);
  mapStatusWarnVFATPerLayer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapStatusErrVFATPerLayer_.SetBinConfX(
      nNewNumCh, stationInfo.nMinIdxChamber_ - 0.5, stationInfo.nMaxIdxChamber_ + 0.5);
  mapStatusErrVFATPerLayer_.SetBinConfY(stationInfo.nMaxVFAT_, -0.5);
  mapStatusErrVFATPerLayer_.bookND(bh, key);
  mapStatusErrVFATPerLayer_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);
  mapStatusErrVFATPerLayer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  return 0;
}

int GEMDAQStatusSource::ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo &stationInfo = mapStationInfo_[key3];

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/VFATStatus_" + getNameDirLayer(key3));

  mapStatusVFATPerCh_.SetBinConfX(stationInfo.nMaxVFAT_, -0.5);
  mapStatusVFATPerCh_.bookND(bh, key);
  mapStatusVFATPerCh_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 1);
  if (mapStatusVFATPerCh_.isOperating()) {
    SetLabelVFATStatus(mapStatusVFATPerCh_.FindHist(key));
  }

  bh.getBooker()->setCurrentFolder(strFolderMain_);

  return 0;
}

void GEMDAQStatusSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  edm::Handle<GEMVFATStatusCollection> gemVFAT;
  edm::Handle<GEMOHStatusCollection> gemOH;
  edm::Handle<GEMAMCStatusCollection> gemAMC;
  edm::Handle<GEMAMC13StatusCollection> gemAMC13;

  event.getByToken(tagVFAT_, gemVFAT);
  event.getByToken(tagOH_, gemOH);
  event.getByToken(tagAMC_, gemAMC);
  event.getByToken(tagAMC13_, gemAMC13);

  if (!(gemVFAT.isValid() && gemOH.isValid() && gemAMC.isValid() && gemAMC13.isValid())) {
    if (!bWarnedNotFound_) {
      edm::LogWarning(log_category_) << "DAQ sources from muonGEMDigis are not found";
      bWarnedNotFound_ = true;
    }
    return;
  }

  std::map<ME4IdsKey, bool> mapChamberAll;
  std::map<ME4IdsKey, bool> mapChamberWarning;
  std::map<ME4IdsKey, bool> mapChamberError;
  std::map<ME4IdsKey, bool> mapChamberVFATWarning;
  std::map<ME4IdsKey, bool> mapChamberVFATError;
  std::map<ME4IdsKey, bool> mapChamberOHWarning;
  std::map<ME4IdsKey, bool> mapChamberOHError;
  std::map<ME4IdsKey, bool> mapChamberAMCWarning;
  std::map<ME4IdsKey, bool> mapChamberAMCError;
  std::map<ME4IdsKey, bool> mapChamberAMC13Error;

  for (auto amc13It = gemAMC13->begin(); amc13It != gemAMC13->end(); ++amc13It) {
    int fedId = (*amc13It).first;
    if (mapFEDIdToPosition_.find(fedId) == mapFEDIdToPosition_.end()) {
      continue;
    }
    int nXBin = mapFEDIdToPosition_[fedId];

    const auto &range = (*amc13It).second;
    for (auto amc13 = range.first; amc13 != range.second; ++amc13) {
      Bool_t bWarn = false;
      Bool_t bErr = false;

      GEMAMC13Status::Warnings warnings{amc13->warnings()};
      GEMAMC13Status::Errors errors{amc13->errors()};

      if (bFillAMC_) {
        if (nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          if (warnings.InValidAMC)
            FillWithRiseErr(h2AMC13Status_, nXBin, 2, bWarn);
        }

        if (nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          if (errors.InValidSize)
            FillWithRiseErr(h2AMC13Status_, nXBin, 3, bErr);
          if (errors.failTrailerCheck)
            FillWithRiseErr(h2AMC13Status_, nXBin, 4, bErr);
          if (errors.failFragmentLength)
            FillWithRiseErr(h2AMC13Status_, nXBin, 5, bErr);
          if (errors.failTrailerMatch)
            FillWithRiseErr(h2AMC13Status_, nXBin, 6, bErr);
          if (errors.moreTrailers)
            FillWithRiseErr(h2AMC13Status_, nXBin, 7, bErr);
          if (errors.crcModified)
            FillWithRiseErr(h2AMC13Status_, nXBin, 8, bErr);
          if (errors.slinkError)
            FillWithRiseErr(h2AMC13Status_, nXBin, 9, bErr);
          if (errors.wrongFedId)
            FillWithRiseErr(h2AMC13Status_, nXBin, 10, bErr);
        }
      }

      if (!bWarn && !bErr) {
        if (bFillAMC_ && nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          h2AMC13Status_->Fill(nXBin, 1);
        }
      } else {
        auto &listChamber = mapAMC13ToListChamber_[fedId];
        for (auto gid : listChamber) {
          ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
          if (bErr)
            mapChamberAMC13Error[key4Ch] = false;
        }
      }
    }
  }

  MonitorElement *h2AMCStatus = nullptr;

  for (auto amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt) {
    int fedId = (*amcIt).first;
    if (mapFEDIdToAMCStatus_.find(fedId) == mapFEDIdToAMCStatus_.end()) {
      continue;
    }
    h2AMCStatus = mapFEDIdToAMCStatus_[fedId];

    const GEMAMCStatusCollection::Range &range = (*amcIt).second;
    for (auto amc = range.first; amc != range.second; ++amc) {
      Bool_t bWarn = false;
      Bool_t bErr = false;

      Int_t nAMCNum = amc->amcNumber();

      GEMAMCStatus::Warnings warnings{amc->warnings()};
      GEMAMCStatus::Errors errors{amc->errors()};

      if (bFillAMC_) {
        if (nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          if (warnings.InValidOH)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 2, bWarn);
          if (warnings.backPressure)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 3, bWarn);
        }

        if (nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          if (errors.badEC)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 4, bErr);
          if (errors.badBC)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 5, bErr);
          if (errors.badOC)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 6, bErr);
          if (errors.badRunType)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 7, bErr);
          if (errors.badCRC)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 8, bErr);
          if (errors.MMCMlocked)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 9, bErr);
          if (errors.DAQclocklocked)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 10, bErr);
          if (errors.DAQnotReday)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 11, bErr);
          if (errors.BC0locked)
            FillWithRiseErr(h2AMCStatus, nAMCNum, 12, bErr);
        }
      }

      if (!bWarn && !bErr) {
        if (bFillAMC_ && nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
          h2AMCStatus->Fill(nAMCNum, 1);
        }
      } else {
        auto &listChamber = mapAMCToListChamber_[{fedId, nAMCNum}];
        for (auto gid : listChamber) {
          ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
          if (bErr)
            mapChamberAMCError[key4Ch] = false;
          if (bWarn)
            mapChamberAMCWarning[key4Ch] = false;
        }
      }
    }
  }

  for (auto ohIt = gemOH->begin(); ohIt != gemOH->end(); ++ohIt) {
    GEMDetId gid = (*ohIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta

    const GEMOHStatusCollection::Range &range = (*ohIt).second;
    for (auto OHStatus = range.first; OHStatus != range.second; ++OHStatus) {
      GEMOHStatus::Warnings warnings{OHStatus->warnings()};
      if (warnings.EvtNF)
        mapStatusOH_.Fill(key3, gid.chamber(), 2);
      if (warnings.InNF)
        mapStatusOH_.Fill(key3, gid.chamber(), 3);
      if (warnings.L1aNF)
        mapStatusOH_.Fill(key3, gid.chamber(), 4);
      if (warnings.EvtSzW)
        mapStatusOH_.Fill(key3, gid.chamber(), 5);
      if (warnings.InValidVFAT)
        mapStatusOH_.Fill(key3, gid.chamber(), 6);

      GEMOHStatus::Errors errors{OHStatus->errors()};
      if (errors.EvtF)
        mapStatusOH_.Fill(key3, gid.chamber(), 7);
      if (errors.InF)
        mapStatusOH_.Fill(key3, gid.chamber(), 8);
      if (errors.L1aF)
        mapStatusOH_.Fill(key3, gid.chamber(), 9);
      if (errors.EvtSzOFW)
        mapStatusOH_.Fill(key3, gid.chamber(), 10);
      if (errors.Inv)
        mapStatusOH_.Fill(key3, gid.chamber(), 11);
      if (errors.OOScAvV)
        mapStatusOH_.Fill(key3, gid.chamber(), 12);
      if (errors.OOScVvV)
        mapStatusOH_.Fill(key3, gid.chamber(), 13);
      if (errors.BxmAvV)
        mapStatusOH_.Fill(key3, gid.chamber(), 14);
      if (errors.BxmVvV)
        mapStatusOH_.Fill(key3, gid.chamber(), 15);
      if (errors.InUfw)
        mapStatusOH_.Fill(key3, gid.chamber(), 16);
      if (errors.badVFatCount)
        mapStatusOH_.Fill(key3, gid.chamber(), 17);

      Bool_t bWarn = warnings.wcodes != 0;
      Bool_t bErr = errors.codes != 0;
      if (!bWarn && !bErr)
        mapStatusOH_.Fill(key3, gid.chamber(), 1);
      if (bWarn)
        mapChamberOHWarning[key4] = false;
      if (bErr)
        mapChamberOHError[key4] = false;
      mapChamberAll[key4] = true;
    }
  }

  for (auto vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt) {
    GEMDetId gid = (*vfatIt).first;
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};  // WARNING: Chamber, not iEta
    const GEMVFATStatusCollection::Range &range = (*vfatIt).second;

    for (auto vfatStat = range.first; vfatStat != range.second; ++vfatStat) {
      Int_t nIdxVFAT = getVFATNumber(gid.station(), gid.ieta(), vfatStat->vfatPosition());

      GEMVFATStatus::Warnings warnings{vfatStat->warnings()};
      if (warnings.basicOFW)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 2);
      if (warnings.zeroSupOFW)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 3);

      GEMVFATStatus::Errors errors{(uint8_t)vfatStat->errors()};
      mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 4);
      if (errors.InValidHeader)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 5);
      if (errors.EC)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 6);
      if (errors.BC)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 7);

      Bool_t bWarn = warnings.wcodes != 0;
      Bool_t bErr = errors.codes != 0;
      if (!bWarn && !bErr)
        mapStatusVFATPerCh_.Fill(key4Ch, nIdxVFAT, 1);
      if (bWarn)
        mapChamberVFATWarning[key4Ch] = false;
      if (bErr)
        mapChamberVFATError[key4Ch] = false;
      if (bWarn)
        mapStatusWarnVFATPerLayer_.Fill(key3, gid.chamber(), nIdxVFAT);
      if (bErr)
        mapStatusErrVFATPerLayer_.Fill(key3, gid.chamber(), nIdxVFAT);
      mapChamberAll[key4Ch] = true;
    }
  }

  if (nRunType_ == GEMDQM_RUNTYPE_ALLPLOTS || nRunType_ == GEMDQM_RUNTYPE_ONLINE) {
    // Summarizing all presence of status of each chamber
    for (auto const &[key4, bErr] : mapChamberAll) {
      ME3IdsKey key3 = key4Tokey3(key4);
      Int_t nChamber = keyToChamber(key4);
      h2SummaryStatusAll->Fill(nChamber, mapStationToIdx_[key3]);
    }

    // Summarizing all presence of status of each chamber
    FillStatusSummaryPlot(mapChamberAll, h2SummaryStatusAll);
    // Summarizing all the error and warning occupancy
    FillStatusSummaryPlot(mapChamberVFATWarning, h2SummaryStatusVFATWarning, &mapChamberWarning);
    FillStatusSummaryPlot(mapChamberVFATError, h2SummaryStatusVFATError, &mapChamberError);
    FillStatusSummaryPlot(mapChamberOHWarning, h2SummaryStatusOHWarning, &mapChamberWarning);
    FillStatusSummaryPlot(mapChamberOHError, h2SummaryStatusOHError, &mapChamberError);
    FillStatusSummaryPlot(mapChamberAMCWarning, h2SummaryStatusAMCWarning, &mapChamberWarning);
    FillStatusSummaryPlot(mapChamberAMCError, h2SummaryStatusAMCError, &mapChamberError);
    FillStatusSummaryPlot(mapChamberAMC13Error, h2SummaryStatusAMC13Error, &mapChamberError);

    FillStatusSummaryPlot(mapChamberWarning, h2SummaryStatusWarning);
    FillStatusSummaryPlot(mapChamberError, h2SummaryStatusError);
  }
}

DEFINE_FWK_MODULE(GEMDAQStatusSource);
