#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentSource.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

GEMEffByGEMCSCSegmentSource::GEMEffByGEMCSCSegmentSource(const edm::ParameterSet& ps)
    : GEMDQMEfficiencySourceBase(ps),
      kGEMGeometryTokenBeginRun_(esConsumes<edm::Transition::BeginRun>()),
      kGEMCSCSegmentCollectionToken_(
          consumes<GEMCSCSegmentCollection>(ps.getUntrackedParameter<edm::InputTag>("gemcscSegmentTag"))),
      kMuonCollectionToken_(consumes<reco::MuonCollection>(ps.getUntrackedParameter<edm::InputTag>("muonTag"))),
      kMinCSCRecHits_(ps.getUntrackedParameter<int>("minCSCRecHits")),
      kModeDev_(ps.getUntrackedParameter<bool>("modeDev")),
      kUseMuonSegment_(ps.getUntrackedParameter<bool>("useMuonSegment")),
      kFolder_(ps.getUntrackedParameter<std::string>("folder")) {}

GEMEffByGEMCSCSegmentSource::~GEMEffByGEMCSCSegmentSource() {}

void GEMEffByGEMCSCSegmentSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // GEMDQMEfficiencySourceBase
  desc.addUntracked<edm::InputTag>("ohStatusTag", edm::InputTag("muonGEMDigis", "OHStatus"));
  desc.addUntracked<edm::InputTag>("vfatStatusTag", edm::InputTag("muonGEMDigis", "VFATStatus"));
  desc.addUntracked<bool>("monitorGE11", true);
  desc.addUntracked<bool>("monitorGE21", false);
  desc.addUntracked<bool>("monitorGE0", false);
  desc.addUntracked<bool>("maskChamberWithError", false);
  desc.addUntracked<std::string>("logCategory", "GEMEffByGEMCSCSegmentSource");

  // GEMEffByGEMCSCSegmentSource
  desc.addUntracked<edm::InputTag>("gemcscSegmentTag", edm::InputTag("gemcscSegments"));
  desc.addUntracked<edm::InputTag>("muonTag", edm::InputTag("muons"));
  desc.addUntracked<int>("minCSCRecHits", 6);
  desc.addUntracked<bool>("useMuonSegment", false);
  desc.addUntracked<bool>("modeDev", false);
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/GEMCSCSegment");

  descriptions.addWithDefaultLabel(desc);
}

void GEMEffByGEMCSCSegmentSource::bookHistograms(DQMStore::IBooker& ibooker,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& setup) {
  const GEMGeometry* gem = nullptr;
  if (auto handle = setup.getHandle(kGEMGeometryTokenBeginRun_)) {
    gem = handle.product();
  } else {
    edm::LogError(kLogCategory_ + "|bookHistograms") << "failed to get GEMGeometry";
    return;
  }

  ibooker.setCurrentFolder(kFolder_);

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    if (skipGEMStation(station_id))
      continue;

    if (station_id == 1) {
      ////////////////////////////////////////////////////////////////////////////
      // Region - Station - Layer
      ////////////////////////////////////////////////////////////////////////////
      const std::vector<const GEMSuperChamber*> superchamber_vec = station->superChambers();
      if (not checkRefs(superchamber_vec)) {
        edm::LogError(kLogCategory_) << "got an invalid ptr from GEMStation::superChambers";
        return;
      }

      const std::vector<const GEMChamber*> chamber_vec = superchamber_vec.front()->chambers();
      if (not checkRefs(chamber_vec)) {
        edm::LogError(kLogCategory_) << "got an invalid ptr from GEMSuperChamber::chambers";
        return;
      }

      // we actually loop over layers
      for (const GEMChamber* chamber : chamber_vec) {
        const int layer_id = chamber->id().layer();
        const GEMDetId key = getReStLaKey(chamber->id());
        const TString suffix = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        const TString title = GEMUtils::getSuffixTitle(region_id, station_id, layer_id);

        // book MEs for the efficiency vs the GEM chambver id
        me_chamber_[key] = bookChamber(ibooker, "chamber" + suffix, title, station);
        me_chamber_matched_[key] = bookNumerator1D(ibooker, me_chamber_[key]);

        if (kUseMuonSegment_) {
          me_chamber_muon_segment_[key] = bookChamber(ibooker, "muon_chamber" + suffix, title, station);
          me_chamber_muon_segment_matched_[key] = bookNumerator1D(ibooker, me_chamber_muon_segment_[key]);
        }

        if (kModeDev_) {
          // book MEs for the efficiency vs the number of CSC hits in a CSC segment
          // CSCSegAlgoRU: min hits = 4, max hits = CSC layers = 6
          me_num_csc_hits_[key] = ibooker.book1D("num_csc_hits" + suffix, title, 4, 2.5, 6.5);
          me_num_csc_hits_[key]->setAxisTitle("Number of CSC Hits", 1);
          me_num_csc_hits_matched_[key] = bookNumerator1D(ibooker, me_num_csc_hits_[key]);

          me_csc_reduced_chi2_[key] = ibooker.book1D("reduced_chi2" + suffix, title, 100, 0.0, 10.0);
          me_csc_reduced_chi2_[key]->setAxisTitle("#chi^{2} / N_{dof} of CSC Segment", 1);
          me_csc_reduced_chi2_matched_[key] = bookNumerator1D(ibooker, me_csc_reduced_chi2_[key]);

          me_csc_chamber_type_[key] = bookCSCChamberType(ibooker, "csc_chamber_type" + suffix, title);
          me_csc_chamber_type_matched_[key] = bookNumerator1D(ibooker, me_csc_chamber_type_[key]);
        }
      }  // GEMChamber
    } else {
      edm::LogWarning(kLogCategory_) << "The monitoring for ";  // TODO
      continue;
    }
  }  // GEMStataion
}

// https://github.com/cms-sw/cmssw/blob/CMSSW_12_3_0_pre5/DataFormats/MuonDetId/interface/CSCDetId.h#L187-L193
dqm::impl::MonitorElement* GEMEffByGEMCSCSegmentSource::bookCSCChamberType(DQMStore::IBooker& ibooker,
                                                                           const TString& name,
                                                                           const TString& title) {
  MonitorElement* monitor_element = ibooker.book1D(name, title, 10, 0.5, 10.5);
  monitor_element->setAxisTitle("CSC chamber type", 1);
  for (int chamber_type = 1; chamber_type <= 10; chamber_type++) {
    const std::string label = CSCDetId::chamberName(chamber_type);
    monitor_element->setBinLabel(chamber_type, label, 1);
  }
  return monitor_element;
}

void GEMEffByGEMCSCSegmentSource::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  //////////////////////////////////////////////////////////////////////////////
  // get data from event
  //////////////////////////////////////////////////////////////////////////////
  const GEMCSCSegmentCollection* gemcsc_segment_collection = nullptr;
  if (const edm::Handle<GEMCSCSegmentCollection> handle = event.getHandle(kGEMCSCSegmentCollectionToken_)) {
    gemcsc_segment_collection = handle.product();

  } else {
    edm::LogError(kLogCategory_) << "invalid GEMCSCSegmentCollection";
    return;
  }

  const reco::MuonCollection* muon_collection = nullptr;
  if (kUseMuonSegment_) {
    if (const edm::Handle<reco::MuonCollection> handle = event.getHandle(kMuonCollectionToken_)) {
      muon_collection = handle.product();

    } else {
      edm::LogError(kLogCategory_) << "invalid reco::MuonCollection";
      return;
    }
  }

  const GEMOHStatusCollection* oh_status_collection = nullptr;
  const GEMVFATStatusCollection* vfat_status_collection = nullptr;
  if (kMaskChamberWithError_) {
    if (auto handle = event.getHandle(kGEMOHStatusCollectionToken_)) {
      oh_status_collection = handle.product();
    } else {
      edm::LogError(kLogCategory_) << "failed to get OHVFATStatusCollection";
      return;
    }

    if (auto handle = event.getHandle(kGEMVFATStatusCollectionToken_)) {
      vfat_status_collection = handle.product();
    } else {
      edm::LogError(kLogCategory_) << "failed to get GEMVFATStatusCollection";
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // NOTE
  //////////////////////////////////////////////////////////////////////////////
  if (gemcsc_segment_collection->size() < 1) {
    LogDebug(kLogCategory_) << "empty GEMCSCSegment";
    return;
  }

  if (kUseMuonSegment_) {
    findMatchedME11Segments(muon_collection);
  }

  for (edm::OwnVector<GEMCSCSegment>::const_iterator iter = gemcsc_segment_collection->begin();
       iter != gemcsc_segment_collection->end();
       iter++) {
    const GEMCSCSegment& gemcsc_segment = *iter;

    const CSCDetId csc_id = gemcsc_segment.cscDetId();
    if (csc_id.isME11()) {
      analyzeGE11ME11Segment(gemcsc_segment, oh_status_collection, vfat_status_collection);

    } else {
      LogDebug(kLogCategory_) << "skip " << csc_id;
      continue;
    }
  }  // GEMCSCSegment
}

// TODO doc
void GEMEffByGEMCSCSegmentSource::analyzeGE11ME11Segment(const GEMCSCSegment& gemcsc_segment,
                                                         const GEMOHStatusCollection* oh_status_collection,
                                                         const GEMVFATStatusCollection* vfat_status_collection) {
  const GEMRecHit* ge11_hit_layer1 = nullptr;
  const GEMRecHit* ge11_hit_layer2 = nullptr;

  const CSCDetId csc_id = gemcsc_segment.cscDetId();
  for (const GEMRecHit& gem_hit : gemcsc_segment.gemRecHits()) {
    const GEMDetId gem_id = gem_hit.gemId();

    if (not gem_id.isGE11()) {
      edm::LogWarning(kLogCategory_) << "CSCSegment is in " << csc_id << " but GEMRecHit is in " << gem_id
                                     << ". skip this GEMCSCSegment."
                                     << "check if RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegAlgoRR.cc has changed.";
      return;
    }

    if (kMaskChamberWithError_) {
      const bool has_error = maskChamberWithError(gem_id.chamberId(), oh_status_collection, vfat_status_collection);
      if (has_error) {
        return;
      }
    }

    const int layer = gem_id.layer();
    if (layer == 1) {
      ge11_hit_layer1 = &gem_hit;

    } else if (layer == 2) {
      ge11_hit_layer2 = &gem_hit;

    } else {
      edm::LogError(kLogCategory_) << "isGE11 but got unexpected layer " << gem_id << ". skip this GEMCSCSegment.";
      return;
    }
  }  // GEMRecHit

  checkCoincidenceGE11(ge11_hit_layer1, ge11_hit_layer2, gemcsc_segment);
  checkCoincidenceGE11(ge11_hit_layer2, ge11_hit_layer1, gemcsc_segment);
}

// TODO doc
void GEMEffByGEMCSCSegmentSource::checkCoincidenceGE11(const GEMRecHit* trigger_layer_hit,
                                                       const GEMRecHit* detection_layer_hit,
                                                       const GEMCSCSegment& gemcsc_segment) {
  if (trigger_layer_hit == nullptr) {
    LogDebug(kLogCategory_) << "trigger_layer_hit is nullptr";
    return;
  }

  const GEMDetId trigger_layer_id = trigger_layer_hit->gemId();
  const int detection_layer = trigger_layer_id.layer() == 1 ? 2 : 1;
  // detection layer key
  // GEMDetId(int region, int ring, int station, int layer, int chamber, int ieta)
  const GEMDetId key{trigger_layer_id.region(), 1, trigger_layer_id.station(), detection_layer, 0, 0};

  const int chamber = trigger_layer_id.chamber();
  const bool is_matched = kUseMuonSegment_ ? isME11SegmentMatched(gemcsc_segment.cscSegment()) : false;

  const int num_csc_hits = static_cast<int>(gemcsc_segment.cscRecHits().size());
  const double reduced_chi2 = gemcsc_segment.chi2() / gemcsc_segment.degreesOfFreedom();
  const int csc_chamber_type = gemcsc_segment.cscDetId().iChamberType();

  if (kModeDev_) {
    fillME(me_num_csc_hits_, key, num_csc_hits);
    fillMEWithinLimits(me_csc_reduced_chi2_, key, reduced_chi2);
    fillME(me_csc_chamber_type_, key, csc_chamber_type);

    if (detection_layer_hit) {
      fillME(me_num_csc_hits_matched_, key, num_csc_hits);
      fillMEWithinLimits(me_csc_reduced_chi2_matched_, key, reduced_chi2);
      fillME(me_csc_chamber_type_matched_, key, csc_chamber_type);
    }
  }

  // TODO add a method
  const bool is_good = num_csc_hits >= kMinCSCRecHits_;
  if (not is_good) {
    return;
  }

  // twofold coincidence rate
  fillME(me_chamber_, key, chamber);
  if (is_matched) {
    fillME(me_chamber_muon_segment_, key, chamber);
  }

  // threefold coincidence rate
  if (detection_layer_hit) {
    fillME(me_chamber_matched_, key, chamber);
    if (is_matched) {
      fillME(me_chamber_muon_segment_matched_, key, chamber);
    }
  }
}

// TODO docs
void GEMEffByGEMCSCSegmentSource::findMatchedME11Segments(const reco::MuonCollection* muon_collection) {
  matched_me11_segment_vector_.clear();

  if (muon_collection == nullptr) {
    // TODO log
    return;
  }

  for (unsigned int idx = 0; idx < muon_collection->size(); idx++) {
    const reco::Muon& muon = muon_collection->at(idx);

    for (const reco::MuonChamberMatch& chamber_match : muon.matches()) {
      if (chamber_match.detector() != MuonSubdetId::CSC) {
        continue;
      }

      const CSCDetId csc_id{chamber_match.id};
      if (not csc_id.isME11()) {
        continue;
      }

      for (const reco::MuonSegmentMatch& segment_match : chamber_match.segmentMatches) {
        if (not segment_match.isMask(reco::MuonSegmentMatch::BestInStationByDR)) {
          continue;
        }
        matched_me11_segment_vector_.push_back(segment_match.cscSegmentRef.get());
      }  // MuonSegmentMatch
    }    // MuonChamberMatch
  }      // MuonCollection
}

// TODO
bool GEMEffByGEMCSCSegmentSource::isME11SegmentMatched(const CSCSegment& csc_segment) {
  bool found = false;

  const CSCDetId csc_id = csc_segment.cscDetId();
  if (not csc_id.isME11()) {
    return false;
  }

  for (const CSCSegment* matched_segment : matched_me11_segment_vector_) {
    if (csc_id != matched_segment->cscDetId())
      continue;
    if (csc_segment.localPosition().x() != matched_segment->localPosition().x())
      continue;
    if (csc_segment.localPosition().y() != matched_segment->localPosition().y())
      continue;
    if (csc_segment.localPosition().z() != matched_segment->localPosition().z())
      continue;
    if (csc_segment.time() != matched_segment->time())
      continue;

    found = true;
  }

  return found;
}
