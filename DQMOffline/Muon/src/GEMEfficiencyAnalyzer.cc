#include "DQMOffline/Muon/interface/GEMEfficiencyAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"

GEMEfficiencyAnalyzer::GEMEfficiencyAnalyzer(const edm::ParameterSet& pset) : GEMOfflineDQMBase(pset) {
  rechit_token_ = consumes<GEMRecHitCollection>(pset.getParameter<edm::InputTag>("recHitTag"));
  muon_token_ = consumes<edm::View<reco::Muon> >(pset.getParameter<edm::InputTag>("muonTag"));

  auto muon_service_parameter = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  muon_service_ = new MuonServiceProxy(muon_service_parameter, consumesCollector());

  use_global_muon_ = pset.getUntrackedParameter<bool>("useGlobalMuon");

  residual_x_cut_ = static_cast<float>(pset.getParameter<double>("residualXCut"));

  pt_binning_ = pset.getUntrackedParameter<std::vector<double> >("ptBinning");
  eta_nbins_ = pset.getUntrackedParameter<int>("etaNbins");
  eta_low_ = pset.getUntrackedParameter<double>("etaLow");
  eta_up_ = pset.getUntrackedParameter<double>("etaUp");

  folder_ = pset.getUntrackedParameter<std::string>("folder");

  title_ = (use_global_muon_ ? "Global Muon" : "Standalone Muon");
  matched_title_ = title_ + TString::Format(" (|x_{Muon} - x_{Hit}| < %.1f)", residual_x_cut_);
}

GEMEfficiencyAnalyzer::~GEMEfficiencyAnalyzer() {}

void GEMEfficiencyAnalyzer::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& run,
                                           edm::EventSetup const& isetup) {
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

      const MEMapKey1 key1{region_number, station_number};
      const auto&& station_name_suffix = TString::Format("_ge%s%d1", region_sign, station_number);
      const auto&& station_title_suffix = TString::Format(" : GE %s%d/1", region_sign, station_number);
      bookDetectorOccupancy(ibooker, station, key1, station_name_suffix, station_title_suffix);

      if (station_number == 1) {
        for (const bool is_odd : {true, false}) {
          std::tuple<int, int, bool> key2{region_number, station_number, is_odd};
          const TString&& parity_name_suffix = station_name_suffix + (is_odd ? "_odd" : "_even");
          const TString&& parity_title_suffix =
              station_title_suffix + (is_odd ? ", Odd Superchamber" : ", Even Superchamber");
          bookOccupancy(ibooker, key2, parity_name_suffix, parity_title_suffix);

          for (int ieta = 1; ieta <= GEMeMap::maxEtaPartition_; ieta++) {
            const TString&& ieta_name_suffix = parity_name_suffix + Form("_ieta%d", ieta);
            const TString&& ieta_title_suffix = parity_title_suffix + Form(", i#eta = %d", ieta);
            const MEMapKey3 key3{region_number, station_number, is_odd, ieta};
            bookResolution(ibooker, key3, ieta_name_suffix, ieta_title_suffix);
          }  // ieta
        }    // is_odd

      } else {
        std::tuple<int, int, bool> key2{region_number, station_number, false};
        bookOccupancy(ibooker, key2, station_name_suffix, station_title_suffix);

        for (int ieta = 1; ieta <= GEMeMap::maxEtaPartition_; ieta++) {
          const MEMapKey3 key3{region_number, station_number, false, ieta};
          const TString&& ieta_name_suffix = station_name_suffix + Form("_ieta%d", ieta);
          const TString&& ieta_title_suffix = station_title_suffix + Form(", i#eta = %d", ieta);
          bookResolution(ibooker, key3, ieta_name_suffix, ieta_title_suffix);
        }  // ieta
      }
    }  // station
  }    // region
}

void GEMEfficiencyAnalyzer::bookDetectorOccupancy(DQMStore::IBooker& ibooker,
                                                  const GEMStation* station,
                                                  const MEMapKey1& key,
                                                  const TString& name_suffix,
                                                  const TString& title_suffix) {
  ibooker.setCurrentFolder(folder_ + "/Efficiency");
  BookingHelper helper(ibooker, name_suffix, title_suffix);

  const auto&& superchambers = station->superChambers();
  if (not checkRefs(superchambers)) {
    edm::LogError(log_category_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
    return;
  }

  // the number of GEMChambers per GEMStation
  const int num_ch = superchambers.size() * superchambers.front()->nChambers();
  const int& num_eta = GEMeMap::maxEtaPartition_;

  me_detector_[key] = helper.book2D("detector", title_, num_ch, 0.5, num_ch + 0.5, num_eta, 0.5, num_eta + 0.5);

  me_detector_matched_[key] =
      helper.book2D("detector_matched", matched_title_, num_ch, 0.5, num_ch + 0.5, num_eta, 0.5, num_eta + 0.5);

  setDetLabelsEta(me_detector_[key], station);
  setDetLabelsEta(me_detector_matched_[key], station);
}

void GEMEfficiencyAnalyzer::bookOccupancy(DQMStore::IBooker& ibooker,
                                          const MEMapKey2& key,
                                          const TString& name_suffix,
                                          const TString& title_suffix) {
  ibooker.setCurrentFolder(folder_ + "/Efficiency");
  BookingHelper helper(ibooker, name_suffix, title_suffix);

  me_muon_pt_[key] = helper.book1D("muon_pt", title_, pt_binning_, "Muon p_{T} [GeV]");
  me_muon_eta_[key] = helper.book1D("muon_eta", title_, eta_nbins_, eta_low_, eta_up_, "Muon |#eta|");

  me_muon_pt_matched_[key] = helper.book1D("muon_pt_matched", matched_title_, pt_binning_, "Muon p_{T} [GeV]");
  me_muon_eta_matched_[key] =
      helper.book1D("muon_eta_matched", matched_title_, eta_nbins_, eta_low_, eta_up_, "Muon |#eta|");
}

void GEMEfficiencyAnalyzer::bookResolution(DQMStore::IBooker& ibooker,
                                           const MEMapKey3& key,
                                           const TString& name_suffix,
                                           const TString& title_suffix) {
  ibooker.setCurrentFolder(folder_ + "/Resolution");
  BookingHelper helper(ibooker, name_suffix, title_suffix);

  // NOTE Residual & Pull
  me_residual_x_[key] = helper.book1D("residual_x", title_, 50, -5.0, 5.0, "Residual in Local X [cm]");
  me_residual_y_[key] = helper.book1D("residual_y", title_, 60, -12.0, 12.0, "Residual in Local Y [cm]");
  me_residual_phi_[key] = helper.book1D("residual_phi", title_, 80, -0.008, 0.008, "Residual in Global #phi [rad]");

  me_pull_x_[key] = helper.book1D("pull_x", title_, 60, -3.0, 3.0, "Pull in Local X");
  me_pull_y_[key] = helper.book1D("pull_y", title_, 60, -3.0, 3.0, "Pull in Local Y");
}

void GEMEfficiencyAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<GEMRecHitCollection> rechit_collection;
  event.getByToken(rechit_token_, rechit_collection);
  if (not rechit_collection.isValid()) {
    edm::LogError(log_category_) << "GEMRecHitCollection is invalid" << std::endl;
    return;
  }

  edm::Handle<edm::View<reco::Muon> > muon_view;
  event.getByToken(muon_token_, muon_view);
  if (not muon_view.isValid()) {
    edm::LogError(log_category_) << "View<Muon> is invalid" << std::endl;
  }

  edm::ESHandle<GEMGeometry> gem;
  setup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(log_category_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  edm::ESHandle<TransientTrackBuilder> transient_track_builder;
  setup.get<TransientTrackRecord>().get("TransientTrackBuilder", transient_track_builder);
  if (not transient_track_builder.isValid()) {
    edm::LogError(log_category_) << "TransientTrackRecord is invalid" << std::endl;
    return;
  }

  muon_service_->update(setup);
  edm::ESHandle<Propagator>&& propagator = muon_service_->propagator("SteppingHelixPropagatorAny");
  if (not propagator.isValid()) {
    edm::LogError(log_category_) << "Propagator is invalid" << std::endl;
    return;
  }

  for (const reco::Muon& muon : *muon_view) {
    const reco::Track* track = nullptr;

    if (use_global_muon_ and muon.globalTrack().isNonnull()) {
      track = muon.globalTrack().get();

    } else if ((not use_global_muon_) and muon.outerTrack().isNonnull()) {
      track = muon.outerTrack().get();
    }

    if (track == nullptr) {
      edm::LogError(log_category_) << "failed to get muon track" << std::endl;
      continue;
    }

    const reco::TransientTrack&& transient_track = transient_track_builder->build(track);
    if (not transient_track.isValid()) {
      edm::LogInfo(log_category_) << "failed to build TransientTrack" << std::endl;
      continue;
    }

    for (const GEMEtaPartition* eta_partition : gem->etaPartitions()) {
      // Skip propagation inn the opposite direction.
      if (muon.eta() * eta_partition->id().region() < 0)
        continue;

      const BoundPlane& bound_plane = eta_partition->surface();

      const TrajectoryStateOnSurface&& tsos =
          propagator->propagate(transient_track.outermostMeasurementState(), bound_plane);
      if (not tsos.isValid()) {
        continue;
      }

      const LocalPoint&& tsos_local_pos = tsos.localPosition();
      const LocalPoint tsos_local_pos_2d(tsos_local_pos.x(), tsos_local_pos.y(), 0.0f);
      if (not bound_plane.bounds().inside(tsos_local_pos_2d)) {
        continue;
      }

      const GEMDetId&& gem_id = eta_partition->id();

      bool is_odd = gem_id.station() == 1 ? (gem_id.chamber() % 2 == 1) : false;
      const std::tuple<int, int> key1{gem_id.region(), gem_id.station()};
      const std::tuple<int, int, bool> key2{gem_id.region(), gem_id.station(), is_odd};
      const std::tuple<int, int, bool, int> key3{gem_id.region(), gem_id.station(), is_odd, gem_id.roll()};

      const int chamber_bin = getDetOccXBin(gem_id, gem);

      me_detector_[key1]->Fill(chamber_bin, gem_id.roll());
      me_muon_pt_[key2]->Fill(muon.pt());
      me_muon_eta_[key2]->Fill(std::fabs(muon.eta()));

      const GEMRecHit* matched_hit = findMatchedHit(tsos_local_pos.x(), rechit_collection->get(gem_id));
      if (matched_hit == nullptr) {
        continue;
      }

      me_detector_matched_[key1]->Fill(chamber_bin, gem_id.roll());
      me_muon_pt_matched_[key2]->Fill(muon.pt());
      me_muon_eta_matched_[key2]->Fill(std::fabs(muon.eta()));

      const LocalPoint&& hit_local_pos = matched_hit->localPosition();
      const GlobalPoint&& hit_global_pos = eta_partition->toGlobal(hit_local_pos);
      const GlobalPoint&& tsos_global_pos = tsos.globalPosition();

      const float residual_x = tsos_local_pos.x() - hit_local_pos.x();
      const float residual_y = tsos_local_pos.y() - hit_local_pos.y();
      const float residual_phi = reco::deltaPhi(tsos_global_pos.barePhi(), hit_global_pos.barePhi());

      const LocalError&& tsos_err = tsos.localError().positionError();
      const LocalError&& hit_err = matched_hit->localPositionError();

      const float pull_x = residual_x / std::sqrt(tsos_err.xx() + hit_err.xx());
      const float pull_y = residual_y / std::sqrt(tsos_err.yy() + hit_err.yy());

      me_residual_x_[key3]->Fill(residual_x);
      me_residual_y_[key3]->Fill(residual_y);
      me_residual_phi_[key3]->Fill(residual_phi);

      me_pull_x_[key3]->Fill(pull_x);
      me_pull_y_[key3]->Fill(pull_y);

    }  // GEMChamber
  }    // Muon
}

const GEMRecHit* GEMEfficiencyAnalyzer::findMatchedHit(const float track_local_x,
                                                       const GEMRecHitCollection::range& range) {
  float min_residual_x{residual_x_cut_};
  const GEMRecHit* closest_hit = nullptr;

  for (auto hit = range.first; hit != range.second; ++hit) {
    float residual_x = std::fabs(track_local_x - hit->localPosition().x());
    if (residual_x <= min_residual_x) {
      min_residual_x = residual_x;
      closest_hit = &(*hit);
    }
  }

  return closest_hit;
}
