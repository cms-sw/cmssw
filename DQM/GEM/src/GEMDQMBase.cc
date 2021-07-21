#include "DQM/GEM/interface/GEMDQMBase.h"
#include "Geometry/CommonTopologies/interface/GEMStripTopology.h"

using namespace std;
using namespace edm;

GEMDQMBase::GEMDQMBase(const edm::ParameterSet& cfg) : geomToken_(esConsumes<edm::Transition::BeginRun>()) {
  log_category_ = cfg.getUntrackedParameter<std::string>("logCategory");

  nNumEtaPartitionGE0_ = 0;
  nNumEtaPartitionGE11_ = 0;
  nNumEtaPartitionGE21_ = 0;
}

int GEMDQMBase::initGeometry(edm::EventSetup const& iSetup) {
  GEMGeometry_ = nullptr;
  try {
    //edm::ESHandle<GEMGeometry> hGeom;
    //iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &iSetup.getData(geomToken_);
  } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError(log_category_) << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return -1;
  }

  return 0;
}

// Borrowed from DQMOffline/Muon/src/GEMOfflineDQMBase.cc
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
  gemChambers_.clear();
  const std::vector<const GEMSuperChamber*>& superChambers_ = GEMGeometry_->superChambers();
  for (auto sch : superChambers_) {  // FIXME: This loop can be merged into the below loop
    int n_lay = sch->nChambers();
    for (int l = 0; l < n_lay; l++) {
      Bool_t bExist = false;
      for (const auto& ch : gemChambers_) {
        if (ch.id() == sch->chamber(l + 1)->id()) {
          bExist = true;
          break;
        }
      }
      if (bExist)
        continue;
      gemChambers_.push_back(*sch->chamber(l + 1));
    }
  }

  // Borrwed from DQMOffline/Muon/src/GEMOfflineMonitor.cc
  nMaxNumCh_ = 0;
  for (const GEMRegion* region : GEMGeometry_->regions()) {
    const int region_number = region->region();

    for (const GEMStation* station : region->stations()) {
      const auto&& superchambers = station->superChambers();

      const int station_number = station->station();
      const int num_superchambers = superchambers.size();
      const int num_layers = superchambers.front()->nChambers();
      const int max_vfat = getMaxVFAT(station->station());  // the number of VFATs per GEMEtaPartition
      const int num_etas = getNumEtaPartitions(station);    // the number of eta partitions per GEMChamber
      const int num_vfat = num_etas * max_vfat;             // the number of VFATs per GEMChamber
      const int num_strip = GEMeMap::maxChan_;              // the number of strips (channels) per VFAT

      nMaxNumCh_ = std::max(nMaxNumCh_, num_superchambers);

      for (int layer_number = 1; layer_number <= num_layers; layer_number++) {
        ME3IdsKey key3(region_number, station_number, layer_number);
        mapStationInfo_[key3] = MEStationInfo(
            region_number, station_number, layer_number, num_superchambers, num_etas, num_vfat, num_strip);
      }
    }
  }

  if (mapStationInfo_.find(ME3IdsKey(-1, 0, 1)) != mapStationInfo_.end())
    nNumEtaPartitionGE0_ = mapStationInfo_[ME3IdsKey(-1, 0, 1)].nNumEtaPartitions_;
  if (mapStationInfo_.find(ME3IdsKey(-1, 1, 1)) != mapStationInfo_.end())
    nNumEtaPartitionGE11_ = mapStationInfo_[ME3IdsKey(-1, 1, 1)].nNumEtaPartitions_;
  if (mapStationInfo_.find(ME3IdsKey(-1, 2, 1)) != mapStationInfo_.end())
    nNumEtaPartitionGE21_ = mapStationInfo_[ME3IdsKey(-1, 2, 1)].nNumEtaPartitions_;

  return 0;
}

int GEMDQMBase::SortingLayers(std::vector<ME3IdsKey>& listLayers) {
  std::sort(listLayers.begin(), listLayers.end(), [](ME3IdsKey key1, ME3IdsKey key2) {
    Int_t re1 = std::get<0>(key1), st1 = std::get<1>(key1), la1 = std::get<2>(key1);
    Int_t re2 = std::get<0>(key2), st2 = std::get<1>(key2), la2 = std::get<2>(key2);
    if (re1 < 0 && re2 > 0)
      return false;
    if (re1 > 0 && re2 < 0)
      return true;
    Bool_t bRes = (re1 < 0);  // == re2 < 0
    Int_t sum1 = 256 * std::abs(re1) + 16 * st1 + 1 * la1;
    Int_t sum2 = 256 * std::abs(re2) + 16 * st2 + 1 * la2;
    if (sum1 <= sum2)
      return bRes;
    return !bRes;
  });

  return 0;
}

dqm::impl::MonitorElement* GEMDQMBase::CreateSummaryHist(DQMStore::IBooker& ibooker, TString strName) {
  std::vector<ME3IdsKey> listLayers;
  for (auto const& [key, stationInfo] : mapStationInfo_)
    listLayers.push_back(key);
  SortingLayers(listLayers);
  for (Int_t i = 0; i < (Int_t)listLayers.size(); i++)
    mapStationToIdx_[listLayers[i]] = i + 1;

  auto h2Res =
      ibooker.book2D(strName, "", nMaxNumCh_, 0.5, nMaxNumCh_ + 0.5, listLayers.size(), 0.5, listLayers.size() + 0.5);

  if (h2Res == nullptr)
    return nullptr;

  for (Int_t i = 1; i <= nMaxNumCh_; i++)
    h2Res->setBinLabel(i, Form("%i", i), 1);
  for (Int_t i = 1; i <= (Int_t)listLayers.size(); i++) {
    auto key = listLayers[i - 1];
    auto strInfo = GEMUtils::getSuffixName(key);  // NOTE: It starts with '_'
    auto label = Form("GE%+i/%iL%i;%s", keyToRegion(key), keyToStation(key), keyToLayer(key), strInfo.Data());
    h2Res->setBinLabel(i, label, 2);
    Int_t nNumCh = mapStationInfo_[key].nNumChambers_;
    h2Res->setBinContent(0, i, nNumCh);
  }

  return h2Res;
}

int GEMDQMBase::GenerateMEPerChamber(DQMStore::IBooker& ibooker) {
  MEMap2Check_.clear();
  MEMap3Check_.clear();
  MEMap3WithChCheck_.clear();
  MEMap4Check_.clear();
  for (const auto& ch : gemChambers_) {
    GEMDetId gid = ch.id();
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key3WithChamber{gid.region(), gid.station(), gid.layer(), gid.chamber()};
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
    if (!MEMap3WithChCheck_[key3WithChamber]) {
      auto strSuffixName = GEMUtils::getSuffixName(key3) + Form("_ch%02i", gid.chamber());
      auto strSuffixTitle = GEMUtils::getSuffixTitle(key3) + Form(" Chamber %02i", gid.chamber());
      BookingHelper bh3Ch(ibooker, strSuffixName, strSuffixTitle);
      ProcessWithMEMap3WithChamber(bh3Ch, key3WithChamber);
      MEMap3WithChCheck_[key3WithChamber] = true;
    }
    for (auto iEta : ch.etaPartitions()) {
      GEMDetId rId = iEta->id();
      ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), rId.ieta()};
      if (!MEMap4Check_[key4]) {
        auto strSuffixName = GEMUtils::getSuffixName(key3) + Form("_ieta%02i", rId.ieta());
        auto strSuffixTitle = GEMUtils::getSuffixTitle(key3) + Form(" iEta %02i", rId.ieta());
        BookingHelper bh4(ibooker, strSuffixName, strSuffixTitle);
        ProcessWithMEMap4(bh4, key4);
        MEMap4Check_[key4] = true;
      }
    }
  }
  return 0;
}
