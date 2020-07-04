#include "DQMOffline/Muon/interface/GEMOfflineMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"

GEMOfflineMonitor::GEMOfflineMonitor(const edm::ParameterSet& pset) : GEMOfflineDQMBase(pset) {
  digi_token_ = consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("digiTag"));
  rechit_token_ = consumes<GEMRecHitCollection>(pset.getParameter<edm::InputTag>("recHitTag"));
}

GEMOfflineMonitor::~GEMOfflineMonitor() {}

void GEMOfflineMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("muonGEMDigis"));
  desc.add<edm::InputTag>("recHitTag", edm::InputTag("gemRecHits"));
  desc.addUntracked<std::string>("logCategory", "GEMOfflineMonitor");
  descriptions.add("gemOfflineMonitor", desc);
}

void GEMOfflineMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const& isetup) {
  edm::ESHandle<GEMGeometry> gem;
  isetup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(log_category_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  for (const GEMRegion* region : gem->regions()) {
    const int region_number = region->region();
    const char* region_sign = region_number > 0 ? "+" : "-";

    for (const GEMStation* station : region->stations()) {
      const int station_number = station->station();

      const MEMapKey1 det_key{region_number, station_number};
      const auto&& station_name = TString::Format("_ge%s%d1", region_sign, station_number);
      const auto&& station_title = TString::Format(" : GE %s%d/1", region_sign, station_number);
      bookDetectorOccupancy(ibooker, station, det_key, station_name, station_title);
    }  // station
  }    // region
}

void GEMOfflineMonitor::bookDetectorOccupancy(DQMStore::IBooker& ibooker,
                                              const GEMStation* station,
                                              const MEMapKey1& key,
                                              const TString& name_suffix,
                                              const TString& title_suffix) {
  BookingHelper helper(ibooker, name_suffix, title_suffix);
  const auto&& superchambers = station->superChambers();
  if (not checkRefs(superchambers)) {
    edm::LogError(log_category_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
    return;
  }

  // per station
  const int num_superchambers = superchambers.size();
  const int num_chambers = num_superchambers * superchambers.front()->nChambers();
  // the numer of VFATs per GEMEtaPartition
  const int max_vfat = getMaxVFAT(station->station());
  // the number of eta partitions per GEMChamber
  const int num_etas = getNumEtaPartitions(station);
  // the number of VFATs per GEMChamber
  const int num_vfat = num_etas * max_vfat;

  // NOTE Digi
  ibooker.setCurrentFolder("GEM/GEMOfflineMonitor/Digi");
  me_digi_det_[key] =
      helper.book2D("digi_det", "Digi Occupancy", num_chambers, 0.5, num_chambers + 0.5, num_vfat, 0.5, num_vfat + 0.5);
  setDetLabelsVFAT(me_digi_det_[key], station);

  // NOTE RecHit
  ibooker.setCurrentFolder("GEM/GEMOfflineMonitor/RecHit");
  me_hit_det_[key] =
      helper.book2D("hit_det", "Hit Occupancy", num_chambers, 0.5, num_chambers + 0.5, num_etas, 0.5, num_etas + 0.5);
  setDetLabelsEta(me_hit_det_[key], station);
}

void GEMOfflineMonitor::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<GEMDigiCollection> digi_collection;
  event.getByToken(digi_token_, digi_collection);
  if (not digi_collection.isValid()) {
    edm::LogError(log_category_) << "GEMDigiCollection is invalid!" << std::endl;
    return;
  }

  edm::Handle<GEMRecHitCollection> rechit_collection;
  event.getByToken(rechit_token_, rechit_collection);
  if (not rechit_collection.isValid()) {
    edm::LogError(log_category_) << "GEMRecHitCollection is invalid" << std::endl;
    return;
  }

  edm::ESHandle<GEMGeometry> gem;
  setup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(log_category_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  // GEMDigi
  for (auto range_iter = digi_collection->begin(); range_iter != digi_collection->end(); range_iter++) {
    const GEMDetId& gem_id = (*range_iter).first;
    const GEMDigiCollection::Range& range = (*range_iter).second;

    const MEMapKey1 det_key{gem_id.region(), gem_id.station()};
    for (auto digi = range.first; digi != range.second; ++digi) {
      const int chamber_bin = getDetOccXBin(gem_id, gem);
      const int vfat_number = getVFATNumberByStrip(gem_id.station(), gem_id.roll(), digi->strip());

      fillME(me_digi_det_, det_key, chamber_bin, vfat_number);
    }
  }

  // GEMRecHit
  for (auto hit = rechit_collection->begin(); hit != rechit_collection->end(); hit++) {
    const GEMDetId&& gem_id = hit->gemId();
    const MEMapKey1 det_key{gem_id.region(), gem_id.station()};

    const int chamber_bin = getDetOccXBin(gem_id, gem);
    fillME(me_hit_det_, det_key, chamber_bin, gem_id.roll());
  }
}
