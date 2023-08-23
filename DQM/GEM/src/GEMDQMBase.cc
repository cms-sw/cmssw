#include "DQM/GEM/interface/GEMDQMBase.h"
#include "Geometry/CommonTopologies/interface/GEMStripTopology.h"

using namespace std;
using namespace edm;

GEMDQMBase::GEMDQMBase(const edm::ParameterSet& cfg) : geomToken_(esConsumes<edm::Transition::BeginRun>()) {
  std::string strRunType = cfg.getUntrackedParameter<std::string>("runType");

  nRunType_ = GEMDQM_RUNTYPE_ONLINE;

  if (strRunType == "online") {
    nRunType_ = GEMDQM_RUNTYPE_ONLINE;
  } else if (strRunType == "offline") {
    nRunType_ = GEMDQM_RUNTYPE_OFFLINE;
  } else if (strRunType == "relval") {
    nRunType_ = GEMDQM_RUNTYPE_RELVAL;
  } else if (strRunType == "allplots") {
    nRunType_ = GEMDQM_RUNTYPE_ALLPLOTS;
  } else {
    edm::LogError(log_category_) << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
  }

  log_category_ = cfg.getUntrackedParameter<std::string>("logCategory");
}

int GEMDQMBase::initGeometry(edm::EventSetup const& iSetup) {
  GEMGeometry_ = nullptr;
  if (auto handle = iSetup.getHandle(geomToken_)) {
    GEMGeometry_ = handle.product();
  } else {
    edm::LogError(log_category_) << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return -1;
  }

  return 0;
}

// Borrowed from DQM/GEM/src/GEMOfflineDQMBase.cc
int GEMDQMBase::getNumEtaPartitions(const GEMStation* station) {
  const auto&& superchambers = station->superChambers();
  if (not checkRefs(superchambers)) {
    edm::LogError(log_category_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
    return 0;
  }

  const auto& chambers = superchambers.front()->chambers();
  if (not checkRefs(chambers)) {
    edm::LogError(log_category_) << "failed to get a valid vector of GEMChamber ptrs" << std::endl;
    return 0;
  }

  return chambers.front()->nEtaPartitions();
}

int GEMDQMBase::loadChambers() {
  if (GEMGeometry_ == nullptr)
    return -1;
  listChamberId_.clear();
  mapEtaPartition_.clear();
  for (const GEMRegion* region : GEMGeometry_->regions()) {
    for (const GEMStation* station : region->stations()) {
      for (auto sch : station->superChambers()) {
        for (auto pchamber : sch->chambers()) {
          GEMDetId gid = pchamber->id();
          listChamberId_.push_back(pchamber->id());
          for (auto iEta : pchamber->etaPartitions()) {
            mapEtaPartition_[gid].push_back(iEta);
          }
        }
      }
    }
  }

  // Borrwed from DQM/GEM/src/GEMOfflineMonitor.cc
  nMaxNumCh_ = 0;
  for (const GEMRegion* region : GEMGeometry_->regions()) {
    const int region_number = region->region();

    for (const GEMStation* station : region->stations()) {
      const auto&& superchambers = station->superChambers();

      const int station_number = station->station();
      const int num_superchambers = (station_number == 1 ? 36 : 18);
      const int num_mod = getNumModule(station->station());
      const int max_vfat = getMaxVFAT(station->station());  // the number of VFATs per GEMEtaPartition
      const int num_etas = getNumEtaPartitions(station);    // the number of eta partitions per GEMChamber
      const int num_vfat = num_etas * max_vfat;             // the number of VFATs per GEMChamber
      const int strip1st = (station_number == 2 ? 1 : 0);   // the index of the first strip
      const int num_digi = GEMeMap::maxChan_;               // the number of digis (channels) per VFAT

      nMaxNumCh_ = std::max(nMaxNumCh_, num_superchambers);

      Int_t nMinIdxChamber = 1048576;
      Int_t nMaxIdxChamber = -1048576;
      for (auto sch : superchambers) {
        auto nIdxChamber = sch->chambers().front()->id().chamber();
        if (nMinIdxChamber > nIdxChamber)
          nMinIdxChamber = nIdxChamber;
        if (nMaxIdxChamber < nIdxChamber)
          nMaxIdxChamber = nIdxChamber;
      }

      const auto& chambers = superchambers.front()->chambers();

      for (auto pchamber : chambers) {
        int layer_number = pchamber->id().layer();
        ME3IdsKey key3(region_number, station_number, layer_number);
        mapStationInfo_[key3] = MEStationInfo(region_number,
                                              station_number,
                                              layer_number,
                                              num_superchambers,
                                              num_mod,
                                              num_etas,
                                              num_vfat,
                                              strip1st,
                                              num_digi,
                                              nMinIdxChamber,
                                              nMaxIdxChamber);
        readGeometryRadiusInfoChamber(station, mapStationInfo_[key3]);
        readGeometryPhiInfoChamber(station, mapStationInfo_[key3]);
      }
    }
  }

  return 0;
}

int GEMDQMBase::SortingLayers(std::vector<ME4IdsKey>& listLayers) {
  std::sort(listLayers.begin(), listLayers.end(), [](ME4IdsKey key1, ME4IdsKey key2) {
    Int_t re1 = std::get<0>(key1), re2 = std::get<0>(key2);
    Int_t st1 = std::get<1>(key1), st2 = std::get<1>(key2);
    Int_t la1 = std::get<2>(key1), la2 = std::get<2>(key2);
    Int_t mo1 = std::get<3>(key1), mo2 = std::get<3>(key2);
    if (re1 < 0 && re2 > 0)
      return false;
    if (re1 > 0 && re2 < 0)
      return true;
    Bool_t bRes = (re1 < 0);  // == re2 < 0
    Int_t sum1 = 4096 * std::abs(re1) + 256 * st1 + 16 * la1 + mo1;
    Int_t sum2 = 4096 * std::abs(re2) + 256 * st2 + 16 * la2 + mo2;
    if (sum1 <= sum2)
      return bRes;
    return !bRes;
  });

  return 0;
}

dqm::impl::MonitorElement* GEMDQMBase::CreateSummaryHist(DQMStore::IBooker& ibooker, TString strName) {
  std::vector<ME4IdsKey> listLayers;
  for (auto const& [key3, stationInfo] : mapStationInfo_) {
    for (int module_number = 1; module_number <= stationInfo.nNumModules_; module_number++) {
      ME4IdsKey key4{keyToRegion(key3), keyToStation(key3), keyToLayer(key3), module_number};
      listLayers.push_back(key4);  // Note: Not only count layers but also modules
    }
  }
  SortingLayers(listLayers);
  for (Int_t i = 0; i < (Int_t)listLayers.size(); i++)
    mapStationToIdx_[listLayers[i]] = i + 1;

  auto h2Res =
      ibooker.book2D(strName, "", nMaxNumCh_, 0.5, nMaxNumCh_ + 0.5, listLayers.size(), 0.5, listLayers.size() + 0.5);
  h2Res->setXTitle("Chamber");
  h2Res->setYTitle("Layer");

  if (h2Res == nullptr)
    return nullptr;

  for (Int_t i = 1; i <= nMaxNumCh_; i++)
    h2Res->setBinLabel(i, Form("%i", i), 1);
  for (Int_t i = 1; i <= (Int_t)listLayers.size(); i++) {
    auto key = listLayers[i - 1];
    ME3IdsKey key3 = key4Tokey3(key);

    auto region = keyToRegion(key);
    auto strInfo = GEMUtils::getSuffixName(key3);  // NOTE: It starts with '_'
    if (mapStationInfo_[key3].nNumModules_ > 1) {
      strInfo += Form("-M%i", keyToModule(key));
    }
    auto label = Form("GE%+i1-%cL%i-M%i;%s",
                      region * keyToStation(key),
                      (region > 0 ? 'P' : 'M'),
                      keyToLayer(key),
                      keyToModule(key),
                      strInfo.Data());
    h2Res->setBinLabel(i, label, 2);
    Int_t nNumCh = mapStationInfo_[key3].nNumChambers_;
    h2Res->setBinContent(0, i, nNumCh);
  }

  return h2Res;
}

int GEMDQMBase::GenerateMEPerChamber(DQMStore::IBooker& ibooker) {
  MEMap2Check_.clear();
  MEMap3Check_.clear();
  MEMap4Check_.clear();
  MEMap5Check_.clear();
  MEMap2WithEtaCheck_.clear();
  MEMap2AbsReWithEtaCheck_.clear();
  MEMap4WithChCheck_.clear();
  MEMap5WithChCheck_.clear();
  for (auto gid : listChamberId_) {
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    const auto num_mod = mapStationInfo_[key3].nNumModules_;
    for (int module_number = 1; module_number <= num_mod; module_number++) {
      ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), module_number};
      ME4IdsKey key4WithChamber{gid.region(), gid.station(), gid.layer(), gid.chamber()};
      ME5IdsKey key5WithChamber{gid.region(), gid.station(), gid.layer(), module_number, gid.chamber()};
      if (!MEMap2Check_[key2]) {
        auto strSuffixName = GEMUtils::getSuffixName(key2);
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key2);
        BookingHelper bh2(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap2(bh2, key2);
        MEMap2Check_[key2] = true;
      }
      if (!MEMap3Check_[key3]) {
        auto strSuffixName = GEMUtils::getSuffixName(key3);
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key3);
        BookingHelper bh3(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap3(bh3, key3);
        MEMap3Check_[key3] = true;
      }
      if (!MEMap4Check_[key4]) {
        Int_t nLa = gid.layer();
        TString strSuffixCh = Form("-L%i", nLa);
        if (mapStationInfo_[key3].nNumModules_ > 1)
          strSuffixCh = Form("-L%i-M%i", nLa, module_number);
        auto strSuffixName = GEMUtils::getSuffixName(key2) + strSuffixCh;
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key2) + strSuffixCh;
        BookingHelper bh4(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap4(bh4, key4);
        MEMap4Check_[key4] = true;
      }
      if (!MEMap4WithChCheck_[key4WithChamber]) {
        Int_t nCh = gid.chamber();
        Int_t nLa = gid.layer();
        char cLS = (nCh % 2 == 0 ? 'L' : 'S');  // FIXME: Is it general enough?
        TString strSuffixCh = Form("-%02iL%i-%c", nCh, nLa, cLS);
        auto strSuffixName = GEMUtils::getSuffixName(key2) + strSuffixCh;
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key2) + strSuffixCh;
        BookingHelper bh4Ch(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap4WithChamber(bh4Ch, key4WithChamber);
        MEMap4WithChCheck_[key4WithChamber] = true;
      }
      if (!MEMap5WithChCheck_[key5WithChamber]) {
        Int_t nCh = gid.chamber();
        Int_t nLa = gid.layer();
        char cLS = (nCh % 2 == 0 ? 'L' : 'S');  // FIXME: Is it general enough?
        TString strSuffixCh = Form("-%02iL%i-%c", nCh, nLa, cLS);
        if (mapStationInfo_[key3].nNumModules_ > 1)
          strSuffixCh = Form("-%02iL%i-M%i-%c", nCh, nLa, module_number, cLS);
        auto strSuffixName = GEMUtils::getSuffixName(key2) + strSuffixCh;
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key2) + strSuffixCh;
        BookingHelper bh5Ch(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap5WithChamber(bh5Ch, key5WithChamber);
        MEMap5WithChCheck_[key5WithChamber] = true;
      }
      for (auto iEta : mapEtaPartition_[gid]) {
        GEMDetId eId = iEta->id();
        ME5IdsKey key5{gid.region(), gid.station(), gid.layer(), module_number, eId.ieta()};
        ME3IdsKey key2WithEta{gid.region(), gid.station(), eId.ieta()};
        ME3IdsKey key2AbsReWithEta{std::abs(gid.region()), gid.station(), eId.ieta()};
        if (!MEMap5Check_[key5]) {
          auto strSuffixName = GEMUtils::getSuffixName(key3) + Form("-E%02i", eId.ieta());
          auto strSuffixTitle = GEMUtils::getSuffixTitle(key3) + Form("-E%02i", eId.ieta());
          BookingHelper bh5(ibooker, strSuffixName, strSuffixTitle);
          ProcessWithMEMap5(bh5, key5);
          MEMap5Check_[key5] = true;
        }
        if (!MEMap2WithEtaCheck_[key2WithEta]) {
          auto strSuffixName = GEMUtils::getSuffixName(key2) + Form("-E%02i", eId.ieta());
          auto strSuffixTitle = GEMUtils::getSuffixTitle(key2) + Form("-E%02i", eId.ieta());
          BookingHelper bh3(ibooker, strSuffixName, strSuffixTitle);
          ProcessWithMEMap2WithEta(bh3, key2WithEta);
          MEMap2WithEtaCheck_[key2WithEta] = true;
        }
        if (!MEMap2AbsReWithEtaCheck_[key2AbsReWithEta]) {
          auto strSuffixName = Form("_GE%d1-E%02i", gid.station(), eId.ieta());
          auto strSuffixTitle = Form(" GE%d1-E%02i", gid.station(), eId.ieta());
          BookingHelper bh3(ibooker, strSuffixName, strSuffixTitle);
          ProcessWithMEMap2AbsReWithEta(bh3, key2AbsReWithEta);
          MEMap2AbsReWithEtaCheck_[key2AbsReWithEta] = true;
        }
      }
    }
  }
  return 0;
}

int GEMDQMBase::readGeometryRadiusInfoChamber(const GEMStation* station, MEStationInfo& stationInfo) {
  auto listSuperChambers = station->superChambers();

  Bool_t bDoneEven = false, bDoneOdd = false;

  // Obtaining radius intervals of even/odd chambers
  for (auto superchamber : listSuperChambers) {
    Int_t chamberNo = superchamber->id().chamber();
    if (chamberNo % 2 == 0 && bDoneEven)
      continue;
    if (chamberNo % 2 != 0 && bDoneOdd)
      continue;

    auto& etaPartitions = superchamber->chambers().front()->etaPartitions();

    // A little of additional procedures to list up the radius intervals
    // It would be independent to local direction of chambers and the order of eta partitions
    //   1. Obtain the radius of the middle top/bottom points of the trapezoid
    //   2. Sort these two values and determine which one is the lower/upper one
    //   3. Keep them all and then sort them
    //   4. The intermediate radii are set as the mean of the corresponding values of upper/lowers.
    std::vector<Float_t> listRadiusLower, listRadiusUpper;
    for (auto iEta : etaPartitions) {
      const GEMStripTopology& stripTopology = dynamic_cast<const GEMStripTopology&>(iEta->specificTopology());
      Float_t fHeight = stripTopology.stripLength();
      LocalPoint lp1(0.0, -0.5 * fHeight), lp2(0.0, 0.5 * fHeight);
      auto& surface = iEta->surface();
      GlobalPoint gp1 = surface.toGlobal(lp1), gp2 = surface.toGlobal(lp2);
      Float_t fR1 = gp1.perp(), fR2 = gp2.perp();
      Float_t fRL = std::min(fR1, fR2), fRH = std::max(fR1, fR2);
      listRadiusLower.push_back(fRL);
      listRadiusUpper.push_back(fRH);
      // For a future usage
      //std::cout << "GEO_RADIUS: " << iEta->id().chamber() << " " << iEta->id().ieta() << " "
      //  << fRL << " " << fRH << std::endl;
    }

    std::sort(listRadiusLower.begin(), listRadiusLower.end());
    std::sort(listRadiusUpper.begin(), listRadiusUpper.end());

    std::vector<Float_t>& listR =
        (chamberNo % 2 == 0 ? stationInfo.listRadiusEvenChamber_ : stationInfo.listRadiusOddChamber_);
    listR.clear();
    listR.push_back(listRadiusLower.front());
    for (int i = 1; i < (int)listRadiusLower.size(); i++) {
      listR.push_back(0.5 * (listRadiusLower[i] + listRadiusUpper[i - 1]));
    }
    listR.push_back(listRadiusUpper.back());

    if (chamberNo % 2 == 0)
      bDoneEven = true;
    if (chamberNo % 2 != 0)
      bDoneOdd = true;

    if (bDoneEven && bDoneOdd)
      break;
  }

  return 0;
}

int GEMDQMBase::readGeometryPhiInfoChamber(const GEMStation* station, MEStationInfo& stationInfo) {
  auto listSuperChambers = station->superChambers();
  Int_t nNumStripEta = stationInfo.nNumDigi_ * (stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_);

  std::vector<std::pair<Int_t, std::pair<std::pair<Float_t, Float_t>, Bool_t>>> listDivPhi;

  // Obtaining phi intervals of chambers
  for (auto superchamber : listSuperChambers) {
    auto iEta = superchamber->chambers().front()->etaPartitions().front();

    // What is the index of the first strip? Rather than to ask to someone, let's calculate it!
    Float_t fWidthStrip = std::abs(iEta->centreOfStrip((Int_t)1).x() - iEta->centreOfStrip((Int_t)0).x());
    LocalPoint lpRef(-fWidthStrip / 3.0, 0.0);
    Int_t nStripMid = (Int_t)iEta->strip(lpRef);
    Int_t nFirstStrip = 1 - ((nNumStripEta / 2) - nStripMid);
    Int_t nLastStrip = nFirstStrip + nNumStripEta - 1;

    auto& surface = iEta->surface();
    LocalPoint lpF = iEta->centreOfStrip((Float_t)(nFirstStrip - 0.5));  // To avoid the round error(?)
    LocalPoint lpL = iEta->centreOfStrip((Float_t)(nLastStrip + 0.5));   // To avoid the round error(?)
    GlobalPoint gpF = surface.toGlobal(lpF);
    GlobalPoint gpL = surface.toGlobal(lpL);

    Float_t fPhiF = gpF.phi();
    Float_t fPhiL = gpL.phi();
    if (fPhiF * fPhiL < 0 && std::abs(fPhiF) > 0.5 * 3.14159265359) {
      if (fPhiF < 0)
        fPhiF += 2 * 3.14159265359;
      if (fPhiL < 0)
        fPhiL += 2 * 3.14159265359;
    }
    Bool_t bFlipped = fPhiF > fPhiL;
    Float_t fPhiMin = std::min(fPhiF, fPhiL);
    Float_t fPhiMax = std::max(fPhiF, fPhiL);

    listDivPhi.emplace_back();
    listDivPhi.back().first = iEta->id().chamber();
    listDivPhi.back().second.first.first = fPhiMin;
    listDivPhi.back().second.first.second = fPhiMax;
    listDivPhi.back().second.second = bFlipped;
  }

  stationInfo.fMinPhi_ = 0.0;
  for (auto p : listDivPhi) {
    if (p.first == 1) {
      stationInfo.fMinPhi_ = p.second.first.first;
      break;
    }
  }

  // For a future usage
  //for ( auto p : listDivPhi ) {
  //  std::cout << "GEO_PHI: " << p.first << " "
  //    << p.second.first.first << " " << p.second.first.second << " " << p.second.second << std::endl;
  //}

  return 0;
}
