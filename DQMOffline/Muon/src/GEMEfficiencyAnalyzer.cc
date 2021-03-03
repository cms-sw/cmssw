#include "DQMOffline/Muon/interface/GEMEfficiencyAnalyzer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonHits/interface/MuonHitHelper.h"

GEMEfficiencyAnalyzer::GEMEfficiencyAnalyzer(const edm::ParameterSet& pset) : GEMOfflineDQMBase(pset) {
  name_ = pset.getUntrackedParameter<std::string>("name");
  folder_ = pset.getUntrackedParameter<std::string>("folder");

  rechit_token_ = consumes<GEMRecHitCollection>(pset.getParameter<edm::InputTag>("recHitTag"));
  muon_token_ = consumes<edm::View<reco::Muon> >(pset.getParameter<edm::InputTag>("muonTag"));

  is_cosmics_ = pset.getUntrackedParameter<bool>("isCosmics");
  use_global_muon_ = pset.getUntrackedParameter<bool>("useGlobalMuon");
  use_skip_layer_ = pset.getParameter<bool>("useSkipLayer");
  use_only_me11_ = pset.getParameter<bool>("useOnlyME11");
  residual_rphi_cut_ = static_cast<float>(pset.getParameter<double>("residualRPhiCut"));
  use_prop_r_error_cut_ = pset.getParameter<bool>("usePropRErrorCut");
  prop_r_error_cut_ = pset.getParameter<double>("propRErrorCut");
  use_prop_phi_error_cut_ = pset.getParameter<bool>("usePropPhiErrorCut");
  prop_phi_error_cut_ = pset.getParameter<double>("propPhiErrorCut");

  pt_bins_ = pset.getUntrackedParameter<std::vector<double> >("ptBins");
  eta_nbins_ = pset.getUntrackedParameter<int>("etaNbins");
  eta_low_ = pset.getUntrackedParameter<double>("etaLow");
  eta_up_ = pset.getUntrackedParameter<double>("etaUp");

  const edm::ParameterSet&& muon_service_parameter = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  muon_service_ = new MuonServiceProxy(muon_service_parameter, consumesCollector());

  const double eps = std::numeric_limits<double>::epsilon();
  pt_clamp_max_ = pt_bins_.back() - eps;
  eta_clamp_max_ = eta_up_ - eps;
}

GEMEfficiencyAnalyzer::~GEMEfficiencyAnalyzer() {}

void GEMEfficiencyAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // beam scenario
  {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::string>("name", "GlobalMuon");
    desc.addUntracked<std::string>("folder", "GEM/Efficiency/type0");
    desc.add<edm::InputTag>("recHitTag", edm::InputTag("gemRecHits"));
    desc.add<edm::InputTag>("muonTag", edm::InputTag("muons"));
    desc.addUntracked<bool>("isCosmics", false);
    desc.addUntracked<bool>("useGlobalMuon", true);
    desc.add<bool>("useSkipLayer", false);
    desc.add<bool>("useOnlyME11", false);
    desc.add<double>("residualRPhiCut", 2.0);  // TODO need to be tuned
    desc.add<bool>("usePropRErrorCut", false);
    desc.add<double>("propRErrorCut", 1.0);
    desc.add<bool>("usePropPhiErrorCut", false);
    desc.add<double>("propPhiErrorCut", 0.01);
    desc.addUntracked<std::vector<double> >("ptBins", {20., 30., 40., 50., 60., 70., 80., 90., 100., 120.});
    desc.addUntracked<int>("etaNbins", 9);
    desc.addUntracked<double>("etaLow", 1.4);
    desc.addUntracked<double>("etaUp", 2.3);
    {
      edm::ParameterSetDescription psd0;
      psd0.setAllowAnything();
      desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
    }
    descriptions.add("gemEfficiencyAnalyzerDefault", desc);
  }  // beam scenario

  // cosmic scenario
  {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::string>("name", "GlobalMuon");  // FIXME
    desc.addUntracked<std::string>("folder", "GEM/Efficiency/type0");
    desc.add<edm::InputTag>("recHitTag", edm::InputTag("gemRecHits"));
    desc.add<edm::InputTag>("muonTag", edm::InputTag("muons"));
    desc.addUntracked<bool>("isCosmics", true);
    desc.addUntracked<bool>("useGlobalMuon", false);
    desc.add<bool>("useSkipLayer", false);
    desc.add<bool>("useOnlyME11", true);
    desc.add<double>("residualRPhiCut", 5.0);  // TODO need to be tuned
    desc.add<bool>("usePropRErrorCut", true);
    desc.add<double>("propRErrorCut", 1.0);
    desc.add<bool>("usePropPhiErrorCut", true);
    desc.add<double>("propPhiErrorCut", 0.001);
    desc.addUntracked<std::vector<double> >(
        "ptBins", {0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 220.});
    desc.addUntracked<int>("etaNbins", 9);
    desc.addUntracked<double>("etaLow", 1.4);
    desc.addUntracked<double>("etaUp", 2.3);
    {
      edm::ParameterSetDescription psd0;
      psd0.setAllowAnything();
      desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
    }
    descriptions.add("gemEfficiencyAnalyzerCosmicsDefault", desc);
  }  // cosmics
}

void GEMEfficiencyAnalyzer::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& run,
                                           edm::EventSetup const& isetup) {
  edm::ESHandle<GEMGeometry> gem;
  isetup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(kLogCategory_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  bookEfficiencyMomentum(ibooker, gem);
  bookEfficiencyChamber(ibooker, gem);
  bookEfficiencyEtaPartition(ibooker, gem);
  bookResolution(ibooker, gem);
  bookMisc(ibooker, gem);
}

dqm::impl::MonitorElement* GEMEfficiencyAnalyzer::bookNumerator1D(DQMStore::IBooker& ibooker, MonitorElement* me) {
  const std::string name = me->getName() + "_matched";
  TH1F* hist = dynamic_cast<TH1F*>(me->getTH1F()->Clone(name.c_str()));
  return ibooker.book1D(name, hist);
}

dqm::impl::MonitorElement* GEMEfficiencyAnalyzer::bookNumerator2D(DQMStore::IBooker& ibooker, MonitorElement* me) {
  const std::string name = me->getName() + "_matched";
  TH2F* hist = dynamic_cast<TH2F*>(me->getTH2F()->Clone(name.c_str()));
  return ibooker.book2D(name, hist);
}

void GEMEfficiencyAnalyzer::bookEfficiencyMomentum(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  // TODO Efficiency/Source
  ibooker.setCurrentFolder(folder_ + "/Efficiency");

  const TString pt_x_title = "Muon p_{T} [GeV]";
  const int pt_nbinsx = pt_bins_.size() - 1;

  const std::string eta_x_title = "Muon |#eta|";
  const std::string phi_x_title = "Muon #phi [rad]";

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const GEMDetId&& key = getReStKey(region_id, station_id);
    const TString&& name_suffix = getSuffixName(region_id, station_id);
    const TString&& title_suffix = getSuffixTitle(region_id, station_id);

    const TString&& title = name_.c_str() + title_suffix;

    TH1F* h_muon_pt = new TH1F("muon_pt" + name_suffix, title, pt_nbinsx, &pt_bins_[0]);
    h_muon_pt->SetXTitle(pt_x_title);
    me_muon_pt_[key] = ibooker.book1D(h_muon_pt->GetName(), h_muon_pt);
    me_muon_pt_matched_[key] = bookNumerator1D(ibooker, me_muon_pt_[key]);

    me_muon_eta_[key] = ibooker.book1D("muon_eta" + name_suffix, title, eta_nbins_, eta_low_, eta_up_);
    me_muon_eta_[key]->setXTitle(eta_x_title);
    me_muon_eta_matched_[key] = bookNumerator1D(ibooker, me_muon_eta_[key]);

    me_muon_phi_[key] = ibooker.book1D("muon_phi" + name_suffix, title, 108, -M_PI, M_PI);
    me_muon_phi_[key]->setAxisTitle(phi_x_title);
    me_muon_phi_matched_[key] = bookNumerator1D(ibooker, me_muon_phi_[key]);
  }  // station
}

void GEMEfficiencyAnalyzer::bookEfficiencyChamber(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  // TODO Efficiency/Source
  ibooker.setCurrentFolder(folder_ + "/Efficiency");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const std::vector<const GEMSuperChamber*>&& superchambers = station->superChambers();
    if (not checkRefs(superchambers)) {
      edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
      return;
    }

    const int num_chambers = superchambers.size();
    for (const GEMChamber* chamber : superchambers[0]->chambers()) {
      const int layer_id = chamber->id().layer();

      const TString&& name_suffix = getSuffixName(region_id, station_id, layer_id);
      const TString&& title_suffix = getSuffixTitle(region_id, station_id, layer_id);
      const GEMDetId&& key = getReStLaKey(chamber->id());

      me_chamber_[key] =
          ibooker.book1D("chamber" + name_suffix, name_.c_str() + title_suffix, num_chambers, 0.5, num_chambers + 0.5);
      me_chamber_[key]->setAxisTitle("Chamber");
      me_chamber_[key]->getTH1F()->SetNdivisions(-num_chambers, "Y");
      for (int binx = 1; binx <= num_chambers; binx++) {
        me_chamber_[key]->setBinLabel(binx, std::to_string(binx));
      }

      me_chamber_matched_[key] = bookNumerator1D(ibooker, me_chamber_[key]);
    }  // layer
  }    // station
}

void GEMEfficiencyAnalyzer::bookEfficiencyEtaPartition(DQMStore::IBooker& ibooker,
                                                       const edm::ESHandle<GEMGeometry>& gem) {
  // TODO Efficiency/Source
  ibooker.setCurrentFolder(folder_ + "/Efficiency");

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const GEMDetId&& key = getReStKey(region_id, station_id);
    const TString&& name_suffix = getSuffixName(region_id, station_id);
    const TString&& title_suffix = getSuffixTitle(region_id, station_id);

    const std::vector<const GEMSuperChamber*>&& superchambers = station->superChambers();
    if (not checkRefs(superchambers)) {
      edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
      return;
    }

    const int num_ch = superchambers.size() * superchambers.front()->nChambers();
    const int num_etas = getNumEtaPartitions(station);

    me_detector_[key] = ibooker.book2D("detector" + name_suffix,
                                       name_.c_str() + title_suffix,
                                       num_ch,
                                       0.5,
                                       num_ch + 0.5,
                                       num_etas,
                                       0.5,
                                       num_etas + 0.5);
    setDetLabelsEta(me_detector_[key], station);
    me_detector_matched_[key] = bookNumerator2D(ibooker, me_detector_[key]);
  }  // station
}

void GEMEfficiencyAnalyzer::bookResolution(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder(folder_ + "/Resolution");
  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const std::vector<const GEMSuperChamber*>&& superchambers = station->superChambers();
    if (not checkRefs(superchambers)) {
      edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
      return;
    }

    const std::vector<const GEMChamber*>& chambers = superchambers[0]->chambers();
    if (not checkRefs(chambers)) {
      edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMChamber ptrs" << std::endl;
      return;
    }

    for (const GEMEtaPartition* eta_partition : chambers[0]->etaPartitions()) {
      const int ieta = eta_partition->id().roll();

      const GEMDetId&& key = getReStEtKey(eta_partition->id());
      // TODO
      const TString&& name_suffix = TString::Format("_GE%+.2d_R%d", region_id * (station_id * 10 + 1), ieta);
      const TString&& title =
          name_.c_str() + TString::Format(" : GE%+.2d Roll %d", region_id * (station_id * 10 + 1), ieta);

      me_residual_rphi_[key] =
          ibooker.book1D("residual_rphi" + name_suffix, title, 50, -residual_rphi_cut_, residual_rphi_cut_);
      me_residual_rphi_[key]->setAxisTitle("Residual in R#phi [cm]");

      me_residual_y_[key] = ibooker.book1D("residual_y" + name_suffix, title, 60, -12.0, 12.0);
      me_residual_y_[key]->setAxisTitle("Residual in Local Y [cm]");

      me_pull_y_[key] = ibooker.book1D("pull_y" + name_suffix, title, 60, -3.0, 3.0);
      me_pull_y_[key]->setAxisTitle("Pull in Local Y");
    }  // ieta
  }    // station
}

void GEMEfficiencyAnalyzer::bookMisc(DQMStore::IBooker& ibooker, const edm::ESHandle<GEMGeometry>& gem) {
  ibooker.setCurrentFolder(folder_ + "/Misc");
  me_prop_r_err_ = ibooker.book1D("prop_r_err", ";Propagation Global R Error [cm];Entries", 20, 0.0, 20.0);
  me_prop_phi_err_ = ibooker.book1D("prop_phi_err", ";Propagation Global Phi Error [rad];Entries", 20, 0.0, M_PI);
  me_all_abs_residual_rphi_ = ibooker.book1D("all_abs_residual_rphi", ";Residual in R#phi [cm];Entries", 20, 0.0, 20.0);

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    const std::vector<const GEMSuperChamber*>&& superchambers = station->superChambers();
    if (not checkRefs(superchambers)) {
      edm::LogError(kLogCategory_) << "failed to get a valid vector of GEMSuperChamber ptrs" << std::endl;
      return;
    }
    // ignore layer ids
    const int num_ch = superchambers.size();

    const GEMDetId&& key = getReStKey(region_id, station_id);
    const TString&& name_suffix = getSuffixName(region_id, station_id);
    const TString&& title_suffix = getSuffixTitle(region_id, station_id);
    me_prop_chamber_[key] = ibooker.book1D("prop_chamber" + name_suffix, title_suffix, num_ch, 0.5, num_ch + 0.5);
    me_prop_chamber_[key]->setAxisTitle("Destination Chamber Id", 1);
    me_prop_chamber_[key]->setAxisTitle("Entries", 2);
  }  // station
}

void GEMEfficiencyAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<GEMRecHitCollection> rechit_collection;
  event.getByToken(rechit_token_, rechit_collection);
  if (not rechit_collection.isValid()) {
    edm::LogError(kLogCategory_) << "GEMRecHitCollection is invalid" << std::endl;
    return;
  }

  edm::Handle<edm::View<reco::Muon> > muon_view;
  event.getByToken(muon_token_, muon_view);
  if (not muon_view.isValid()) {
    edm::LogError(kLogCategory_) << "View<Muon> is invalid" << std::endl;
    return;
  }

  edm::ESHandle<GEMGeometry> gem;
  setup.get<MuonGeometryRecord>().get(gem);
  if (not gem.isValid()) {
    edm::LogError(kLogCategory_) << "GEMGeometry is invalid" << std::endl;
    return;
  }

  edm::ESHandle<GlobalTrackingGeometry> global_tracking_geometry;
  setup.get<GlobalTrackingGeometryRecord>().get(global_tracking_geometry);
  if (not global_tracking_geometry.isValid()) {
    edm::LogError(kLogCategory_) << "GlobalTrackingGeometry is invalid" << std::endl;
    return;
  }

  edm::ESHandle<TransientTrackBuilder> transient_track_builder;
  setup.get<TransientTrackRecord>().get("TransientTrackBuilder", transient_track_builder);
  if (not transient_track_builder.isValid()) {
    edm::LogError(kLogCategory_) << "TransientTrackRecord is invalid" << std::endl;
    return;
  }

  muon_service_->update(setup);
  edm::ESHandle<Propagator>&& propagator = muon_service_->propagator("SteppingHelixPropagatorAny");
  if (not propagator.isValid()) {
    edm::LogError(kLogCategory_) << "Propagator is invalid" << std::endl;
    return;
  }

  if (rechit_collection->size() < 1) {
    edm::LogInfo(kLogCategory_) << "empty rechit collection" << std::endl;
    return;
  }

  if (muon_view->empty()) {
    edm::LogInfo(kLogCategory_) << "empty muon collection" << std::endl;
    return;
  }

  const std::vector<GEMLayerData>&& layer_vector = buildGEMLayers(gem);

  for (const reco::Muon& muon : *muon_view) {
    const reco::Track* track = getTrack(muon);
    if (track == nullptr) {
      edm::LogError(kLogCategory_) << "failed to get a muon track" << std::endl;
      continue;
    }

    const reco::TransientTrack&& transient_track = transient_track_builder->build(track);
    if (not transient_track.isValid()) {
      edm::LogError(kLogCategory_) << "failed to build TransientTrack" << std::endl;
      continue;
    }

    for (const GEMLayerData& layer : layer_vector) {
      if (use_skip_layer_ and skipLayer(track, layer)) {
        edm::LogInfo(kLogCategory_) << "skip GEM Layer" << std::endl;
        continue;
      }

      const auto&& [start_state, start_id] = getStartingState(transient_track, layer, global_tracking_geometry);

      if (not start_state.isValid()) {
        edm::LogInfo(kLogCategory_) << "failed to get a starting state" << std::endl;
        continue;
      }

      if (use_only_me11_ and (not isME11(start_id))) {
        edm::LogInfo(kLogCategory_) << "skip a starting state because it is not ME11" << std::endl;
        continue;
      }

      // the trajectory state on the destination surface
      const TrajectoryStateOnSurface&& dest_state = propagator->propagate(start_state, *(layer.surface));
      if (not dest_state.isValid()) {
        edm::LogInfo(kLogCategory_) << "failed to propagate a muon" << std::endl;
        continue;
      }

      const GlobalPoint&& dest_global_pos = dest_state.globalPosition();

      if (not checkBounds(dest_global_pos, (*layer.surface))) {
        edm::LogInfo(kLogCategory_) << "failed to pass checkBounds" << std::endl;
        continue;
      }

      const GEMEtaPartition* eta_partition = findEtaPartition(dest_global_pos, layer.chambers);
      if (eta_partition == nullptr) {
        edm::LogInfo(kLogCategory_) << "failed to find an eta partition" << std::endl;
        continue;
      }

      const BoundPlane& surface = eta_partition->surface();

      const LocalPoint&& dest_local_pos = eta_partition->toLocal(dest_global_pos);
      const LocalError&& dest_local_err = dest_state.localError().positionError();
      const GlobalError& dest_global_err = ErrorFrameTransformer().transform(dest_local_err, surface);

      const double dest_global_r_err = std::sqrt(dest_global_err.rerr(dest_global_pos));
      const double dest_global_phi_err = std::sqrt(dest_global_err.phierr(dest_global_pos));

      const GEMDetId&& gem_id = eta_partition->id();
      const GEMDetId&& rs_key = getReStKey(gem_id);
      const GEMDetId&& rsl_key = getReStLaKey(gem_id);
      const GEMDetId&& rse_key = getReStEtKey(gem_id);

      const int chamber_bin = getDetOccXBin(gem_id, gem);
      const int ieta = gem_id.roll();

      // FIXME clever way to clamp values?
      me_prop_r_err_->Fill(std::min(dest_global_r_err, 19.999));
      me_prop_phi_err_->Fill(std::min(dest_global_r_err, M_PI - 0.0001));
      me_prop_chamber_[rs_key]->Fill(gem_id.chamber());

      if (use_prop_r_error_cut_ and (dest_global_r_err > prop_r_error_cut_)) {
        edm::LogInfo(kLogCategory_) << "failed to pass a propagation global R error cut" << std::endl;
        continue;
      }

      if (use_prop_phi_error_cut_ and (dest_global_phi_err > prop_phi_error_cut_)) {
        edm::LogInfo(kLogCategory_) << "failed to pass a propagation global phi error cut" << std::endl;
        continue;
      }

      const double muon_pt = std::min(muon.pt(), pt_clamp_max_);
      const double muon_eta = std::clamp(std::fabs(muon.eta()), eta_low_, eta_clamp_max_);

      fillME(me_detector_, rs_key, chamber_bin, ieta);
      fillME(me_muon_pt_, rs_key, muon_pt);
      fillME(me_muon_eta_, rs_key, muon_eta);
      fillME(me_muon_phi_, rs_key, muon.phi());
      fillME(me_chamber_, rsl_key, gem_id.chamber());

      const auto&& [hit, residual_rphi] = findClosetHit(dest_global_pos, rechit_collection->get(gem_id), eta_partition);

      if (hit == nullptr) {
        edm::LogInfo(kLogCategory_) << "failed to find a hit" << std::endl;
        continue;
      }

      me_all_abs_residual_rphi_->Fill(std::min(std::abs(residual_rphi), 19.999f));

      if (std::abs(residual_rphi) > residual_rphi_cut_) {
        edm::LogInfo(kLogCategory_) << "failed to pass the residual rphi cut" << std::endl;
        continue;
      }

      fillME(me_detector_matched_, rs_key, chamber_bin, ieta);
      fillME(me_muon_pt_matched_, rs_key, muon_pt);
      fillME(me_muon_eta_matched_, rs_key, muon_eta);
      fillME(me_muon_phi_matched_, rs_key, muon.phi());
      fillME(me_chamber_matched_, rsl_key, gem_id.chamber());

      const LocalPoint&& hit_local_pos = hit->localPosition();
      const LocalError&& hit_local_err = hit->localPositionError();

      const float residual_y = dest_local_pos.y() - hit_local_pos.y();
      const float pull_y = residual_y / std::sqrt(dest_local_err.yy() + hit_local_err.yy());

      fillME(me_residual_rphi_, rse_key, residual_rphi);
      fillME(me_residual_y_, rse_key, residual_y);
      fillME(me_pull_y_, rse_key, pull_y);
    }  // layer
  }    // Muon
}

std::vector<GEMEfficiencyAnalyzer::GEMLayerData> GEMEfficiencyAnalyzer::buildGEMLayers(
    const edm::ESHandle<GEMGeometry>& gem) {
  std::vector<GEMLayerData> layer_vector;

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    // layer_id - chambers
    std::map<int, std::vector<const GEMChamber*> > chambers_per_layer;  // chambers per layer
    for (const GEMSuperChamber* super_chamber : station->superChambers()) {
      for (const GEMChamber* chamber : super_chamber->chambers()) {
        const int layer_id = chamber->id().layer();

        if (chambers_per_layer.find(layer_id) == chambers_per_layer.end())
          chambers_per_layer.insert({layer_id, std::vector<const GEMChamber*>()});

        chambers_per_layer[layer_id].push_back(chamber);
      }  // GEMChamber
    }    // GEMSuperChamber

    for (auto [layer_id, chamber_vector] : chambers_per_layer) {
      auto [rmin, rmax] = chamber_vector[0]->surface().rSpan();
      auto [zmin, zmax] = chamber_vector[0]->surface().zSpan();
      for (const GEMChamber* chamber : chamber_vector) {
        // the span of a bound surface in the global coordinates
        const auto [chamber_rmin, chamber_rmax] = chamber->surface().rSpan();
        const auto [chamber_zmin, chamber_zmax] = chamber->surface().zSpan();

        rmin = std::min(rmin, chamber_rmin);
        rmax = std::max(rmax, chamber_rmax);

        zmin = std::min(zmin, chamber_zmin);
        zmax = std::max(zmax, chamber_zmax);
      }

      // layer position and rotation
      const float layer_z = chamber_vector[0]->position().z();
      Surface::PositionType position(0.f, 0.f, layer_z);
      Surface::RotationType rotation;

      zmin -= layer_z;
      zmax -= layer_z;

      // the bounds from min and max R and Z in the local coordinates.
      SimpleDiskBounds* bounds = new SimpleDiskBounds(rmin, rmax, zmin, zmax);
      const Disk::DiskPointer&& layer = Disk::build(position, rotation, bounds);

      layer_vector.emplace_back(layer, chamber_vector, region_id, station_id, layer_id);

    }  // layer
  }    // GEMStation

  return layer_vector;
}

const reco::Track* GEMEfficiencyAnalyzer::getTrack(const reco::Muon& muon) {
  const reco::Track* track = nullptr;

  if (is_cosmics_) {
    if (muon.outerTrack().isNonnull())
      track = muon.outerTrack().get();

  } else {
    // beams, i.e. pp or heavy ions
    if (use_global_muon_ and muon.globalTrack().isNonnull())
      track = muon.globalTrack().get();

    else if ((not use_global_muon_) and muon.outerTrack().isNonnull())
      track = muon.outerTrack().get();
  }

  return track;
}

std::pair<TrajectoryStateOnSurface, DetId> GEMEfficiencyAnalyzer::getStartingState(
    const reco::TransientTrack& transient_track,
    const GEMLayerData& layer,
    const edm::ESHandle<GlobalTrackingGeometry>& geometry) {
  TrajectoryStateOnSurface starting_state;
  DetId starting_id;

  if (use_global_muon_) {
    std::tie(starting_state, starting_id) = findStartingState(transient_track, layer, geometry);

  } else {
    // if outer track
    const reco::Track& track = transient_track.track();
    const bool is_insideout = isInsideOut(track);

    const DetId inner_id{(is_insideout ? track.outerDetId() : track.innerDetId())};
    if (MuonHitHelper::isGEM(inner_id.rawId())) {
      std::tie(starting_state, starting_id) = findStartingState(transient_track, layer, geometry);

    } else {
      starting_id = inner_id;
      if (is_insideout)
        starting_state = transient_track.outermostMeasurementState();
      else
        starting_state = transient_track.innermostMeasurementState();
    }
  }

  return std::make_pair(starting_state, starting_id);
}

std::pair<TrajectoryStateOnSurface, DetId> GEMEfficiencyAnalyzer::findStartingState(
    const reco::TransientTrack& transient_track,
    const GEMLayerData& layer,
    const edm::ESHandle<GlobalTrackingGeometry>& geometry) {
  GlobalPoint starting_point;
  DetId starting_id;
  float min_distance = 1e12;
  bool found = false;

  // TODO optimize this loop because hits should be ordered..
  for (auto rechit = transient_track.recHitsBegin(); rechit != transient_track.recHitsEnd(); rechit++) {
    const DetId&& det_id = (*rechit)->geographicalId();

    if (MuonHitHelper::isGEM(det_id.rawId())) {
      continue;
    }

    const GeomDet* det = geometry->idToDet(det_id);
    const GlobalPoint&& global_position = det->toGlobal((*rechit)->localPosition());
    const float distance = std::abs(layer.surface->localZclamped(global_position));
    if (distance < min_distance) {
      found = true;
      min_distance = distance;
      starting_point = global_position;
      starting_id = det_id;
    }
  }

  TrajectoryStateOnSurface starting_state;
  if (found) {
    starting_state = transient_track.stateOnSurface(starting_point);
  }
  return std::make_pair(starting_state, starting_id);
}

bool GEMEfficiencyAnalyzer::isME11(const DetId& det_id) {
  if (not MuonHitHelper::isCSC(det_id))
    return false;
  const CSCDetId csc_id{det_id};
  return (csc_id.station() == 1) or ((csc_id.ring() == 1) or (csc_id.ring() == 4));
}

bool GEMEfficiencyAnalyzer::skipLayer(const reco::Track* track, const GEMLayerData& layer) {
  const bool is_same_region = track->eta() * layer.region > 0;

  bool skip = false;
  if (is_cosmics_) {
    float p2_in = track->innerMomentum().mag2();
    float p2_out = track->outerMomentum().mag2();
    if (isInsideOut(*track))
      std::swap(p2_in, p2_out);
    const bool is_outgoing = p2_in > p2_out;

    skip = (is_outgoing xor is_same_region);

  } else {
    // beam scenario
    skip = not is_same_region;
  }

  return skip;
}

bool GEMEfficiencyAnalyzer::checkBounds(const GlobalPoint& global_point, const Plane& plane) {
  const LocalPoint&& local_point = plane.toLocal(global_point);
  const LocalPoint local_point_2d(local_point.x(), local_point.y(), 0.0f);
  return plane.bounds().inside(local_point_2d);
}

const GEMEtaPartition* GEMEfficiencyAnalyzer::findEtaPartition(const GlobalPoint& global_point,
                                                               const std::vector<const GEMChamber*>& chamber_vector) {
  const GEMEtaPartition* bound = nullptr;
  for (const GEMChamber* chamber : chamber_vector) {
    if (not checkBounds(global_point, chamber->surface()))
      continue;

    for (const GEMEtaPartition* eta_partition : chamber->etaPartitions()) {
      if (checkBounds(global_point, eta_partition->surface())) {
        bound = eta_partition;
        break;
      }
    }  // GEMEtaPartition
  }    // GEMChamber

  return bound;
}

std::pair<const GEMRecHit*, float> GEMEfficiencyAnalyzer::findClosetHit(const GlobalPoint& dest_global_pos,
                                                                        const GEMRecHitCollection::range& range,
                                                                        const GEMEtaPartition* eta_partition) {
  const StripTopology& topology = eta_partition->specificTopology();
  const LocalPoint&& dest_local_pos = eta_partition->toLocal(dest_global_pos);
  const float dest_local_x = dest_local_pos.x();
  const float dest_local_y = dest_local_pos.y();

  const GEMRecHit* closest_hit = nullptr;
  float min_residual_rphi = 1e6;

  for (auto hit = range.first; hit != range.second; ++hit) {
    const LocalPoint&& hit_local_pos = hit->localPosition();
    const float hit_local_phi = topology.stripAngle(eta_partition->strip(hit_local_pos));

    const float residual_x = dest_local_x - hit_local_pos.x();
    const float residual_y = dest_local_y - hit_local_pos.y();
    const float residual_rphi = std::cos(hit_local_phi) * residual_x + std::sin(hit_local_phi) * residual_y;

    if (std::abs(residual_rphi) < std::abs(min_residual_rphi)) {
      min_residual_rphi = residual_rphi;
      closest_hit = &(*hit);
    }
  }

  return std::make_pair(closest_hit, min_residual_rphi);
}
