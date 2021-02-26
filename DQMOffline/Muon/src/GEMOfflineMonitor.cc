#include "DQMOffline/Muon/interface/GEMOfflineMonitor.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

GEMOfflineMonitor::GEMOfflineMonitor(const edm::ParameterSet& pset) : GEMOfflineDQMBase(pset) {
  digi_token_ = consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("digiTag"));
  rechit_token_ = consumes<GEMRecHitCollection>(pset.getParameter<edm::InputTag>("recHitTag"));
  do_digi_occupancy_ = pset.getUntrackedParameter<bool>("doDigiOccupancy");
  do_hit_occupancy_ = pset.getUntrackedParameter<bool>("doHitOccupancy");
  log_category_ = pset.getUntrackedParameter<std::string>("logCategory");
}

GEMOfflineMonitor::~GEMOfflineMonitor() {}

void GEMOfflineMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("muonGEMDigis"));
  desc.add<edm::InputTag>("recHitTag", edm::InputTag("gemRecHits"));
  desc.addUntracked<std::string>("logCategory", "GEMOfflineMonitor");
  desc.addUntracked<bool>("doDigiOccupancy", true);
  desc.addUntracked<bool>("doHitOccupancy", true);
  descriptions.add("gemOfflineMonitorDefault", desc);
}

void GEMOfflineMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const& setup) {
  edm::ESHandle<GEMGeometry> gem;
  setup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(log_category_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  if (do_digi_occupancy_)
    bookDigiOccupancy(ibooker, gem);

  if (do_hit_occupancy_)
    bookHitOccupancy(ibooker, gem);
}

void GEMOfflineMonitor::bookDigiOccupancy(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder("GEM/GEMOfflineMonitor/DigiOccupancy");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();
    const GEMDetId&& key = getReStKey(region_id, station_id);
    const auto&& name_suffix = getSuffixName(region_id, station_id);
    const auto&& title_suffix = getSuffixTitle(region_id, station_id);

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

    me_digi_det_[key] = ibooker.book2D("digi_det" + name_suffix,
                                       "Digi Occupancy" + title_suffix,
                                       num_chambers,
                                       0.5,
                                       num_chambers + 0.5,
                                       num_vfat,
                                       0.5,
                                       num_vfat + 0.5);
    setDetLabelsVFAT(me_digi_det_[key], station);
  }  // station
}

void GEMOfflineMonitor::bookHitOccupancy(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder("GEM/GEMOfflineMonitor/HitOccupancy");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const GEMDetId&& key = getReStKey(region_id, station_id);
    const auto&& name_suffix = getSuffixName(region_id, station_id);
    const auto&& title_suffix = getSuffixTitle(region_id, station_id);

    const auto&& superchambers = station->superChambers();
    if (not checkRefs(superchambers)) {
      edm::LogError(log_category_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
      return;
    }

    // per station
    const int num_superchambers = superchambers.size();
    const int num_chambers = num_superchambers * superchambers.front()->nChambers();
    // the number of eta partitions per GEMChamber
    const int num_etas = getNumEtaPartitions(station);

    me_hit_det_[key] = ibooker.book2D("hit_det" + name_suffix,
                                      "Hit Occupancy" + title_suffix,
                                      num_chambers,
                                      0.5,
                                      num_chambers + 0.5,
                                      num_etas,
                                      0.5,
                                      num_etas + 0.5);
    setDetLabelsEta(me_hit_det_[key], station);
  }  // station
}

void GEMOfflineMonitor::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<GEMDigiCollection> digi_collection;
  if (do_digi_occupancy_) {
    event.getByToken(digi_token_, digi_collection);
    if (not digi_collection.isValid()) {
      edm::LogError(log_category_) << "GEMDigiCollection is invalid!" << std::endl;
      return;
    }
  }

  edm::Handle<GEMRecHitCollection> rechit_collection;
  if (do_hit_occupancy_) {
    event.getByToken(rechit_token_, rechit_collection);
    if (not rechit_collection.isValid()) {
      edm::LogError(log_category_) << "GEMRecHitCollection is invalid" << std::endl;
      return;
    }
  }

  edm::ESHandle<GEMGeometry> gem;
  setup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(log_category_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  if (do_digi_occupancy_)
    doDigiOccupancy(gem, digi_collection);

  if (do_hit_occupancy_)
    doHitOccupancy(gem, rechit_collection);
}

void GEMOfflineMonitor::doDigiOccupancy(const edm::ESHandle<GEMGeometry>& gem,
                                        const edm::Handle<GEMDigiCollection>& digi_collection) {
  for (auto range_iter = digi_collection->begin(); range_iter != digi_collection->end(); range_iter++) {
    const GEMDetId& gem_id = (*range_iter).first;
    const GEMDigiCollection::Range& range = (*range_iter).second;

    const GEMDetId&& rs_key = getReStKey(gem_id);
    for (auto digi = range.first; digi != range.second; ++digi) {
      const int chamber_bin = getDetOccXBin(gem_id, gem);
      const int vfat_number = getVFATNumberByStrip(gem_id.station(), gem_id.roll(), digi->strip());

      fillME(me_digi_det_, rs_key, chamber_bin, vfat_number);
    }  // digi
  }    // range
}

void GEMOfflineMonitor::doHitOccupancy(const edm::ESHandle<GEMGeometry>& gem,
                                       const edm::Handle<GEMRecHitCollection>& rechit_collection) {
  for (auto hit = rechit_collection->begin(); hit != rechit_collection->end(); hit++) {
    const GEMDetId&& gem_id = hit->gemId();
    const GEMDetId&& rs_key = getReStKey(gem_id);
    const int chamber_bin = getDetOccXBin(gem_id, gem);
    fillME(me_hit_det_, rs_key, chamber_bin, gem_id.roll());
  }
}
