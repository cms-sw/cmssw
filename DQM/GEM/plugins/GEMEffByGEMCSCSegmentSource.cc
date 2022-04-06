#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentSource.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"
#include <algorithm> // std::clamp

GEMEffByGEMCSCSegmentSource::GEMEffByGEMCSCSegmentSource(const edm::ParameterSet& parameter_set)
    : kGEMTokenBeginRun_(esConsumes<edm::Transition::BeginRun>()),
      kGEMCSCSegmentToken_(
          consumes<GEMCSCSegmentCollection>(parameter_set.getParameter<edm::InputTag>("gemcscSegmentTag"))),
      kMuonToken_(consumes<reco::MuonCollection>(parameter_set.getParameter<edm::InputTag>("muonTag"))),
      kUseMuon_(parameter_set.getUntrackedParameter<bool>("useMuon")),
      kMinCSCRecHits_(parameter_set.getUntrackedParameter<uint32_t>("minCSCRecHits")),
      kFolder_(parameter_set.getUntrackedParameter<std::string>("folder")),
      kLogCategory_(parameter_set.getUntrackedParameter<std::string>("logCategory")) {}

GEMEffByGEMCSCSegmentSource::~GEMEffByGEMCSCSegmentSource() {}

void GEMEffByGEMCSCSegmentSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemcscSegmentTag", edm::InputTag("gemcscSegments"));
  desc.add<edm::InputTag>("muonTag", edm::InputTag("muons"));
  desc.addUntracked<bool>("useMuon", false);
  desc.addUntracked<uint32_t>("minCSCRecHits", 6u);
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/GEMCSCSegment");
  desc.addUntracked<std::string>("logCategory", "GEMEffByGEMCSCSegmentSource");
  descriptions.addWithDefaultLabel(desc);
}

void GEMEffByGEMCSCSegmentSource::bookHistograms(DQMStore::IBooker& ibooker,
                                                 edm::Run const&,
                                                 edm::EventSetup const& setup) {
  const edm::ESHandle<GEMGeometry>& gem = setup.getHandle(kGEMTokenBeginRun_);
  if (not gem.isValid()) {
    edm::LogError(kLogCategory_) << "invalid GEMGeometry";
    return;
  }

  bookEfficiencyChamber(ibooker, gem);
  bookMisc(ibooker, gem);
}

void GEMEffByGEMCSCSegmentSource::bookEfficiencyChamber(DQMStore::IBooker& ibooker,
                                                        const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder(kFolder_ + "/Efficiency");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    if (station_id == 1) {
      // GE11
      const std::vector<const GEMSuperChamber*> superchambers = station->superChambers();
      if (not checkRefs(superchambers)) {
        edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs";
        return;
      }

      const int num_chambers = superchambers.size();
      for (const GEMChamber* chamber : superchambers.at(0)->chambers()) {
        const int layer_id = chamber->id().layer();

        const TString name_suffix = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        const TString title_suffix = GEMUtils::getSuffixTitle(region_id, station_id, layer_id);
        const GEMDetId key = getReStLaKey(chamber->id());

        me_chamber_[key] = ibooker.book1D("chamber" + name_suffix, title_suffix, num_chambers, 0.5, num_chambers + 0.5);
        me_chamber_[key]->setAxisTitle("Chamber", 1);
        for (int binx = 1; binx <= num_chambers; binx++) {
          me_chamber_[key]->setBinLabel(binx, std::to_string(binx), 1);
        }
        me_chamber_matched_[key] = bookNumerator1D(ibooker, me_chamber_[key]);

        if (kUseMuon_) {
          me_muon_chamber_[key] =
              ibooker.book1D("muon_chamber" + name_suffix, title_suffix, num_chambers, 0.5, num_chambers + 0.5);
          me_muon_chamber_[key]->setAxisTitle("Chamber", 1);
          for (int binx = 1; binx <= num_chambers; binx++) {
            me_muon_chamber_[key]->setBinLabel(binx, std::to_string(binx), 1);
          }
          me_muon_chamber_matched_[key] = bookNumerator1D(ibooker, me_muon_chamber_[key]);
        }
      }  // layer

    } else {
      LogDebug(kLogCategory_) << "skip " << station->getName();
      continue;
    }
  }  // station
}

void GEMEffByGEMCSCSegmentSource::bookMisc(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder(kFolder_ + "/Misc");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    if (station_id == 1) {
      // GE11
      const std::vector<const GEMSuperChamber*> superchambers = station->superChambers();
      if (not checkRefs(superchambers)) {
        edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs";
        return;
      }

      for (const GEMChamber* chamber : superchambers.at(0)->chambers()) {
        const int layer_id = chamber->id().layer();

        const TString name_suffix = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        const TString title_suffix = GEMUtils::getSuffixTitle(region_id, station_id, layer_id);
        const GEMDetId key = getReStLaKey(chamber->id());

        // num_csc_hits
        me_num_csc_hits_[key] = ibooker.book1D("num_csc_hits" + name_suffix, title_suffix, 4, 2.5, 6.5);
        me_num_csc_hits_[key]->setAxisTitle("Number of CSCRecHits", 1);

        me_num_csc_hits_matched_[key] = bookNumerator1D(ibooker, me_num_csc_hits_[key]);

        // reduced_chi2
        me_reduced_chi2_[key] = ibooker.book1D("reduced_chi2" + name_suffix, title_suffix, 30, 0, 3);
        me_reduced_chi2_[key]->setAxisTitle("#chi^{2} / dof", 1);

        me_reduced_chi2_matched_[key] = bookNumerator1D(ibooker, me_reduced_chi2_[key]);

        // CSC chamber type
        // https://github.com/cms-sw/cmssw/blob/CMSSW_12_3_0_pre5/DataFormats/MuonDetId/interface/CSCDetId.h#L187-L193
        me_csc_chamber_type_[key] = ibooker.book1D("csc_chamber_type" + name_suffix, title_suffix, 10, 0.5, 10.5);
        me_csc_chamber_type_[key]->setAxisTitle("CSC chamber type", 1);
        for (int chamber_type = 1; chamber_type <= 10; chamber_type++) {
          const std::string label = CSCDetId::chamberName(chamber_type);
          me_csc_chamber_type_[key]->setBinLabel(chamber_type, label, 1);
        }

        me_csc_chamber_type_matched_[key] = bookNumerator1D(ibooker, me_csc_chamber_type_[key]);

      }  // layer

    } else {
      LogDebug(kLogCategory_) << "skip " << station->getName();
      continue;
    }
  }  // region-station
}

dqm::impl::MonitorElement* GEMEffByGEMCSCSegmentSource::bookNumerator1D(DQMStore::IBooker& ibooker,
                                                                        MonitorElement* me) {
  const std::string name = me->getName() + "_matched";
  TH1F* hist = dynamic_cast<TH1F*>(me->getTH1F()->Clone(name.c_str()));
  return ibooker.book1D(name, hist);
}

void GEMEffByGEMCSCSegmentSource::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  //////////////////////////////////////////////////////////////////////////////
  // get data from Event & EventSetup
  const GEMCSCSegmentCollection* gemcsc_segment_collection = nullptr;
  if (const edm::Handle<GEMCSCSegmentCollection> handle = event.getHandle(kGEMCSCSegmentToken_)) {
    gemcsc_segment_collection = handle.product();

  } else {
    edm::LogError(kLogCategory_) << "invalid GEMCSCSegmentCollection";
    return;
  }

  const reco::MuonCollection* muon_collection = nullptr;
  if (kUseMuon_) {
    if (const edm::Handle<reco::MuonCollection> handle = event.getHandle(kMuonToken_)) {
      muon_collection = handle.product();

    } else {
      edm::LogError(kLogCategory_) << "invalid reco::MuonCollection";
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // quick check
  if (gemcsc_segment_collection->size() < 1) {
    LogDebug(kLogCategory_) << "empty GEMCSCSegment";
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //
  if (kUseMuon_) {
    findMatchedME11Segments(muon_collection);
  }

  //////////////////////////////////////////////////////////////////////////////
  // main loop
  for (edm::OwnVector<GEMCSCSegment>::const_iterator iter = gemcsc_segment_collection->begin();
       iter != gemcsc_segment_collection->end();
       iter++) {
    const GEMCSCSegment& gemcsc_segment = *iter;

    const CSCDetId csc_id = gemcsc_segment.cscDetId();
    if (csc_id.isME11()) {
      analyzeME11GE11Segment(gemcsc_segment);

    } else {
      LogDebug(kLogCategory_) << "skip " << csc_id;
      continue;
    }
  }  // GEMCSCSegment
}

void GEMEffByGEMCSCSegmentSource::analyzeME11GE11Segment(const GEMCSCSegment& gemcsc_segment) {
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
  const bool is_matched = kUseMuon_ ? isME11SegmentMatched(gemcsc_segment.cscSegment()) : false;

  const int num_csc_hits = gemcsc_segment.cscRecHits().size();
  const double reduced_chi2 = gemcsc_segment.chi2() / gemcsc_segment.degreesOfFreedom();
  const int csc_chamber_type = gemcsc_segment.cscDetId().iChamberType();

  // TODO add a method
  const bool is_good = gemcsc_segment.cscRecHits().size() >= kMinCSCRecHits_;

  fillME(me_num_csc_hits_, key, num_csc_hits);
  fillMEWithinLimits(me_reduced_chi2_, key, reduced_chi2);
  fillME(me_csc_chamber_type_, key, csc_chamber_type);
  if (detection_layer_hit) {
    fillME(me_num_csc_hits_matched_, key, num_csc_hits);
    fillMEWithinLimits(me_reduced_chi2_matched_, key, reduced_chi2);
    fillME(me_csc_chamber_type_matched_, key, csc_chamber_type);
  }

  if (is_good) {
    // twofold coincidence rate
    fillME(me_chamber_, key, chamber);
    if (is_matched) {
      fillME(me_muon_chamber_, key, chamber);
    }

    // threefold coincidence rate
    if (detection_layer_hit) {
      fillME(me_chamber_matched_, key, chamber);
      if (is_matched) {
        fillME(me_muon_chamber_matched_, key, chamber);
      }
    }
  }
}

void GEMEffByGEMCSCSegmentSource::findMatchedME11Segments(const reco::MuonCollection* muon_collection) {
  matched_me11_segment_vector_.clear();
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

bool GEMEffByGEMCSCSegmentSource::hasMEKey(const MEMap& me_map, const GEMDetId& key) {
  bool has_key = true;

  if UNLIKELY (me_map.find(key) == me_map.end()) {
    const std::string hint = me_map.empty() ? "empty" : me_map.begin()->second->getName();
    edm::LogError(kLogCategory_) << "got an invalid key: " << key << ", hint=" << hint;
    has_key = false;

  }

  return has_key;
}

void GEMEffByGEMCSCSegmentSource::fillME(dqm::impl::MonitorElement* me, const double x) {
  if (me) {
    me->Fill(x);

  } else {
    edm::LogError(kLogCategory_) << "MonitorElement is nullptr";

  }
}

void GEMEffByGEMCSCSegmentSource::fillME(MEMap& me_map, const GEMDetId& key, const double x) {
  if (hasMEKey(me_map, key)) {
    me_map[key]->Fill(x);
  }
}

// https://github.com/cms-sw/cmssw/blob/CMSSW_12_0_0_pre3/DQMOffline/L1Trigger/src/L1TFillWithinLimits.cc
void GEMEffByGEMCSCSegmentSource::fillMEWithinLimits(dqm::impl::MonitorElement* me, const double x) {
  if (me) {
    const double xlow = me->getAxisMin(1);
    const double xup = me->getAxisMax(1) - kEps_;
    me->Fill(std::clamp(x, xlow, xup));

  } else {
    edm::LogError(kLogCategory_) << "MonitorElement is nullptr";

  }
}

void GEMEffByGEMCSCSegmentSource::fillMEWithinLimits(MEMap& me_map, const GEMDetId& key, const double x) {
  if (hasMEKey(me_map, key)) {
    fillMEWithinLimits(me_map[key], x);
  }
}
