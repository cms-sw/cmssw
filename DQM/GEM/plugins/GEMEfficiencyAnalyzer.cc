#include "DQM/GEM/plugins/GEMEfficiencyAnalyzer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Validation/MuonHits/interface/MuonHitHelper.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

GEMEfficiencyAnalyzer::GEMEfficiencyAnalyzer(const edm::ParameterSet& ps)
    : GEMDQMEfficiencySourceBase(ps),
      kGEMGeometryTokenBeginRun_(esConsumes<edm::Transition::BeginRun>()),
      kTransientTrackBuilderToken_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      kGEMRecHitCollectionToken_(consumes<GEMRecHitCollection>(ps.getUntrackedParameter<edm::InputTag>("recHitTag"))),
      kMuonViewToken_(consumes<edm::View<reco::Muon> >(ps.getUntrackedParameter<edm::InputTag>("muonTag"))),
      kMuonTrackTypeName_(ps.getUntrackedParameter<std::string>("muonTrackType")),
      kMuonTrackType_(getMuonTrackType(kMuonTrackTypeName_)),
      kMuonName_(TString(ps.getUntrackedParameter<std::string>("muonName"))),
      kFolder_(ps.getUntrackedParameter<std::string>("folder")),
      kScenario_(getScenarioOption(ps.getUntrackedParameter<std::string>("scenario"))),
      kStartingStateType_(getStartingStateType(ps.getUntrackedParameter<std::string>("startingStateType"))),
      kMuonSubdetForGEM_({
          ps.getUntrackedParameter<std::vector<int> >("muonSubdetForGE0"),
          ps.getUntrackedParameter<std::vector<int> >("muonSubdetForGE11"),
          ps.getUntrackedParameter<std::vector<int> >("muonSubdetForGE21"),
      }),
      kCSCForGEM_({
          ps.getUntrackedParameter<std::vector<int> >("cscForGE0"),
          ps.getUntrackedParameter<std::vector<int> >("cscForGE11"),
          ps.getUntrackedParameter<std::vector<int> >("cscForGE21"),
      }),
      kMuonSegmentMatchDRCut_(static_cast<float>(ps.getUntrackedParameter<double>("muonSegmentMatchDRCut"))),
      kMuonPtMinCuts_({
          ps.getUntrackedParameter<double>("muonPtMinCutGE0"),
          ps.getUntrackedParameter<double>("muonPtMinCutGE11"),
          ps.getUntrackedParameter<double>("muonPtMinCutGE21"),
      }),
      kMuonEtaMinCuts_({
          ps.getUntrackedParameter<double>("muonEtaMinCutGE0"),
          ps.getUntrackedParameter<double>("muonEtaMinCutGE11"),
          ps.getUntrackedParameter<double>("muonEtaMinCutGE21"),
      }),
      kMuonEtaMaxCuts_({
          ps.getUntrackedParameter<double>("muonEtaMaxCutGE0"),
          ps.getUntrackedParameter<double>("muonEtaMaxCutGE11"),
          ps.getUntrackedParameter<double>("muonEtaMaxCutGE21"),
      }),
      kPropagationErrorRCut_(static_cast<float>(ps.getUntrackedParameter<double>("propagationErrorRCut"))),
      kPropagationErrorPhiCut_(static_cast<float>(ps.getUntrackedParameter<double>("propagationErrorPhiCut"))),
      kBoundsErrorScale_(static_cast<float>(ps.getUntrackedParameter<double>("boundsErrorScale"))),
      kMatchingMetric_(getMatchingMetric(ps.getUntrackedParameter<std::string>("matchingMetric"))),
      kMatchingCut_(static_cast<float>(ps.getUntrackedParameter<double>("matchingCut"))),
      kMuonPtBins_(ps.getUntrackedParameter<std::vector<double> >("muonPtBins")),
      kMuonEtaNbins_({
          ps.getUntrackedParameter<int>("muonEtaNbinsGE0"),
          ps.getUntrackedParameter<int>("muonEtaNbinsGE11"),
          ps.getUntrackedParameter<int>("muonEtaNbinsGE21"),
      }),
      kMuonEtaLow_({
          ps.getUntrackedParameter<double>("muonEtaLowGE0"),
          ps.getUntrackedParameter<double>("muonEtaLowGE11"),
          ps.getUntrackedParameter<double>("muonEtaLowGE21"),
      }),
      kMuonEtaUp_({
          ps.getUntrackedParameter<double>("muonEtaUpGE0"),
          ps.getUntrackedParameter<double>("muonEtaUpGE11"),
          ps.getUntrackedParameter<double>("muonEtaUpGE21"),
      }),
      kModeDev_(ps.getUntrackedParameter<bool>("modeDev")) {
  muon_service_ =
      std::make_unique<MuonServiceProxy>(ps.getParameter<edm::ParameterSet>("ServiceParameters"), consumesCollector());
}

GEMEfficiencyAnalyzer::~GEMEfficiencyAnalyzer() {}

void GEMEfficiencyAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // GEMDQMEfficiencySourceBase
  desc.addUntracked<edm::InputTag>("ohStatusTag", edm::InputTag("muonGEMDigis", "OHStatus"));
  desc.addUntracked<edm::InputTag>("vfatStatusTag", edm::InputTag("muonGEMDigis", "VFATStatus"));
  desc.addUntracked<bool>("monitorGE11", true);
  desc.addUntracked<bool>("monitorGE21", false);
  desc.addUntracked<bool>("monitorGE0", false);
  desc.addUntracked<bool>("maskChamberWithError", false);
  desc.addUntracked<std::string>("logCategory", "GEMEfficiencyAnalyzer");

  // GEMEfficiencyAnalyzer
  desc.addUntracked<edm::InputTag>("recHitTag", edm::InputTag("gemRecHits"));
  desc.addUntracked<edm::InputTag>("muonTag", edm::InputTag("muons"));
  desc.addUntracked<bool>("modeDev", false);
  desc.addUntracked<std::string>("muonTrackType", "OuterTrack");
  desc.addUntracked<std::string>("muonName", "STA Muon");
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/muonSTA");
  desc.addUntracked<std::string>("scenario", "pp");
  //
  desc.addUntracked<std::string>("startingStateType", "OutermostMeasurementState");
  desc.addUntracked<double>("muonSegmentMatchDRCut", 5.0f);  // for cosmics, in cm,  TODO tune
  // muon pt cut
  desc.addUntracked<double>("muonPtMinCutGE0", 20.0f);
  desc.addUntracked<double>("muonPtMinCutGE11", 20.0f);
  desc.addUntracked<double>("muonPtMinCutGE21", 20.0f);
  // muon abs eta cut for GE11
  desc.addUntracked<double>("muonEtaMinCutGE11", 1.5);
  desc.addUntracked<double>("muonEtaMaxCutGE11", 2.2);
  // muon abs eta cut for GE21
  desc.addUntracked<double>("muonEtaMinCutGE21", 1.5);
  desc.addUntracked<double>("muonEtaMaxCutGE21", 2.5);
  // muon abs eta cut for GE0
  desc.addUntracked<double>("muonEtaMinCutGE0", 2.0);
  desc.addUntracked<double>("muonEtaMaxCutGE0", 3.0);
  // propagation error cuts
  desc.addUntracked<double>("propagationErrorRCut", 0.5);    // cm
  desc.addUntracked<double>("propagationErrorPhiCut", 0.2);  // degree
  //
  desc.addUntracked<double>("boundsErrorScale", -2.0);  // TODO tune
  // matching
  desc.addUntracked<std::string>("matchingMetric", "DeltaPhi");
  desc.addUntracked<double>("matchingCut", 0.2);  // DeltaPhi for pp, in degree TODO tune
  // for MinotorElement
  const std::vector<double> default_pt_bins{
      0, 5, 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110.};  // actually edges
  desc.addUntracked<std::vector<double> >("muonPtBins", default_pt_bins);
  // GE11
  desc.addUntracked<int>("muonEtaNbinsGE11", 9);  // bin width = 0.1
  desc.addUntracked<double>("muonEtaLowGE11", 1.4);
  desc.addUntracked<double>("muonEtaUpGE11", 2.3);
  // GE21
  desc.addUntracked<int>("muonEtaNbinsGE21", 12);  // bin width = 0.1
  desc.addUntracked<double>("muonEtaLowGE21", 1.4);
  desc.addUntracked<double>("muonEtaUpGE21", 2.6);
  // GE0
  desc.addUntracked<int>("muonEtaNbinsGE0", 12);  // bin width = 0.1
  desc.addUntracked<double>("muonEtaLowGE0", 1.9);
  desc.addUntracked<double>("muonEtaUpGE0", 3.1);

  // MuonSubdetId's are listed in DataFormats/MuonDetId/interface/MuonSubdetId.h
  desc.addUntracked<std::vector<int> >("muonSubdetForGE0", {});  // allow all muon subdetectors. TODO optimzie.
  desc.addUntracked<std::vector<int> >("muonSubdetForGE11", {});
  desc.addUntracked<std::vector<int> >("muonSubdetForGE21", {});
  // INFO when muonTrackType is "CombinedTrack" or "OuterTrack"
  // https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/DataFormats/MuonDetId/interface/CSCDetId.h#L187-L193
  // assumed to be the same area.
  desc.addUntracked<std::vector<int> >("cscForGE11", {1, 2});  // ME1a, ME1b
  desc.addUntracked<std::vector<int> >("cscForGE21", {});      // all CSCSegments allowed
  desc.addUntracked<std::vector<int> >("cscForGE0", {});       // all CSCSegments allowed

  // ServiceParameters for MuonServiceProxy
  // This will be initialized in the cfi file
  edm::ParameterSetDescription service_parameters;
  service_parameters.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("ServiceParameters", service_parameters);

  descriptions.add("gemEfficiencyAnalyzerDefault", desc);
}

// convert a string to enum
GEMEfficiencyAnalyzer::MatchingMetric GEMEfficiencyAnalyzer::getMatchingMetric(const std::string name) {
  MatchingMetric method;

  if (name == "DeltaPhi") {
    method = MatchingMetric::kDeltaPhi;

  } else if (name == "RdPhi") {
    method = MatchingMetric::kRdPhi;

  } else {
    edm::LogError(kLogCategory_) << "received an unexpected MatchingMetric: " << name
                                 << " -> MatchingMetric::kDeltaPhi will be used instead.";
    method = MatchingMetric::kDeltaPhi;
  }

  return method;
}

// convert a string to enum
GEMEfficiencyAnalyzer::StartingStateType GEMEfficiencyAnalyzer::getStartingStateType(const std::string name) {
  StartingStateType type;

  if (name == "InnermostMeasurementState") {
    type = StartingStateType::kInnermostMeasurementState;

  } else if (name == "OutermostMeasurementState") {
    type = StartingStateType::kOutermostMeasurementState;

  } else if (name == "StateOnSurfaceWithCSCSegment") {
    type = StartingStateType::kStateOnSurfaceWithCSCSegment;

  } else if (name == "AlignmentStyle") {
    type = StartingStateType::kAlignmentStyle;

  } else {
    edm::LogError(kLogCategory_) << "received an unexpected StartingStateType: " << name
                                 << " -> StartingStateType::kOutermostMeasurementState will be used instead.";
    type = StartingStateType::kOutermostMeasurementState;
  }

  return type;
}

// convert a string to enum
reco::Muon::MuonTrackType GEMEfficiencyAnalyzer::getMuonTrackType(const std::string name) {
  reco::Muon::MuonTrackType muon_track_type;

  // DO NOT ALLOW TYPO
  if (name == "InnerTrack") {
    muon_track_type = reco::Muon::MuonTrackType::InnerTrack;

  } else if (name == "OuterTrack") {
    muon_track_type = reco::Muon::MuonTrackType::OuterTrack;

  } else if (name == "CombinedTrack") {
    muon_track_type = reco::Muon::MuonTrackType::CombinedTrack;

  } else {
    edm::LogError(kLogCategory_) << "received an unexpected reco::Muon::MuonTrackType: " << name
                                 << " --> OuterTrack will be used instead.";

    muon_track_type = reco::Muon::MuonTrackType::OuterTrack;
  }

  return muon_track_type;
}

GEMEfficiencyAnalyzer::ScenarioOption GEMEfficiencyAnalyzer::getScenarioOption(const std::string name) {
  ScenarioOption scenario;
  if (name == "pp") {
    scenario = ScenarioOption::kPP;

  } else if (name == "cosmics") {
    scenario = ScenarioOption::kCosmics;

  } else if (name == "HeavyIons") {
    scenario = ScenarioOption::kHeavyIons;

    edm::LogInfo(kLogCategory_) << "The scenario is set to \"HeavyIons\""
                                << " but there is no strategy dedicated to"
                                << "\"HeavyIons\" scenario. The strategy for "
                                << "the \"pp\" scenario will be used insteqad.";

  } else {
    scenario = ScenarioOption::kPP;

    edm::LogError(kLogCategory_) << "received an unexpected ScenarioOption: " << name
                                 << ". Choose from (\"pp\", \"cosmics\", \"HeavyIons\")"
                                 << " --> pp will be used instead.";
  }

  return scenario;
}

void GEMEfficiencyAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& setup) {
  ibooker.setCurrentFolder(kFolder_);

  const GEMGeometry* gem = nullptr;
  if (auto handle = setup.getHandle(kGEMGeometryTokenBeginRun_)) {
    gem = handle.product();
  } else {
    edm::LogError(kLogCategory_ + "|bookHistograms") << "failed to get GEMGeometry";
    return;
  }

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();

    if (skipGEMStation(station_id)) {
      continue;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Region-Station
    ////////////////////////////////////////////////////////////////////////////
    {  // shadowing to reuse short variable names
      const GEMDetId key = getReStKey(region_id, station_id);
      const TString suffix = GEMUtils::getSuffixName(region_id, station_id);
      const TString title = kMuonName_ + GEMUtils::getSuffixTitle(region_id, station_id);

      // sources for eff. vs muon pt
      TH1F* h_muon_pt = new TH1F("muon_pt" + suffix, title, kMuonPtBins_.size() - 1, &kMuonPtBins_[0]);
      me_muon_pt_[key] = ibooker.book1D(h_muon_pt->GetName(), h_muon_pt);
      me_muon_pt_[key]->setAxisTitle("Muon p_{T} [GeV]", 1);
      me_muon_pt_matched_[key] = bookNumerator1D(ibooker, me_muon_pt_[key]);

      // sources for eff. vs muon eta
      me_muon_eta_[key] = ibooker.book1D("muon_eta" + suffix,
                                         title,
                                         kMuonEtaNbins_.at(station_id),
                                         kMuonEtaLow_.at(station_id),
                                         kMuonEtaUp_.at(station_id));
      me_muon_eta_[key]->setAxisTitle("Muon |#eta|", 1);
      me_muon_eta_matched_[key] = bookNumerator1D(ibooker, me_muon_eta_[key]);

      // sources for eff. vs muon phi
      me_muon_phi_[key] = ibooker.book1D("muon_phi" + suffix, title, 36, -180, 180);
      me_muon_phi_[key]->setAxisTitle("Muon #phi [deg]");
      me_muon_phi_matched_[key] = bookNumerator1D(ibooker, me_muon_phi_[key]);

      if (kModeDev_) {
        // without cuts except the fiducial cut
        TH1F* h_muon_pt_all = new TH1F("muon_pt_all" + suffix, title, kMuonPtBins_.size() - 1, &kMuonPtBins_[0]);
        me_muon_pt_all_[key] = ibooker.book1D(h_muon_pt_all->GetName(), h_muon_pt_all);
        me_muon_pt_all_[key]->setAxisTitle("Muon p_{T} [GeV]", 1);
        me_muon_pt_all_matched_[key] = bookNumerator1D(ibooker, me_muon_pt_all_[key]);

        me_muon_eta_all_[key] = ibooker.book1D("muon_eta_all" + suffix,
                                               title,
                                               kMuonEtaNbins_.at(station_id),
                                               kMuonEtaLow_.at(station_id),
                                               kMuonEtaUp_.at(station_id));
        me_muon_eta_all_[key]->setAxisTitle("Muon |#eta|", 1);
        me_muon_eta_all_matched_[key] = bookNumerator1D(ibooker, me_muon_eta_all_[key]);

        me_muon_charge_[key] = ibooker.book1D("muon_charge" + suffix, title, 3, -1.5, 1.5);
        me_muon_charge_[key]->setAxisTitle("Muon charge", 1);
        me_muon_charge_matched_[key] = bookNumerator1D(ibooker, me_muon_charge_[key]);
      }
    }  // shadowing

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

      {  // shadowing
        const GEMDetId key = getReStLaKey(chamber->id());
        const TString suffix = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        const TString title = kMuonName_ + GEMUtils::getSuffixTitle(region_id, station_id, layer_id);

        me_chamber_ieta_[key] = bookChamberEtaPartition(ibooker, "chamber_ieta" + suffix, title, station);
        me_chamber_ieta_matched_[key] = bookNumerator2D(ibooker, me_chamber_ieta_[key]);

        if (kModeDev_) {
          me_prop_path_length_[key] = ibooker.book1D("prop_path_length" + suffix, title, 50, 0.0, 5.0);
          me_prop_path_length_[key]->setAxisTitle("Propagation path length [cm]", 1);
          me_prop_path_length_matched_[key] = bookNumerator1D(ibooker, me_prop_path_length_[key]);

          // prop. r error in the global coordinates
          me_prop_err_r_[key] = ibooker.book1D("prop_err_r" + suffix, title, 60, 0.0, 3.0);
          me_prop_err_r_[key]->setAxisTitle("Propagation global #sigma_{R} [cm]", 1);
          me_prop_err_r_matched_[key] = bookNumerator1D(ibooker, me_prop_err_r_[key]);

          // prop. r error in the global coordinates
          me_prop_err_phi_[key] = ibooker.book1D("prop_err_phi" + suffix, title, 50, 0.0, 1.0);
          me_prop_err_phi_[key]->setAxisTitle("Propagation's global #sigma_{#phi} [deg]", 1);
          me_prop_err_phi_matched_[key] = bookNumerator1D(ibooker, me_prop_err_phi_[key]);

          // cutflow
          me_cutflow_[key] = ibooker.book1D("cutflow" + suffix, title, 5, 0.5, 5.5);
          me_cutflow_[key]->setBinLabel(1, "All");
          me_cutflow_[key]->setBinLabel(2, Form("#sigma_{R} < %.3f cm", kPropagationErrorRCut_));
          me_cutflow_[key]->setBinLabel(3, Form("#sigma_{phi} < %.3f deg", kPropagationErrorPhiCut_));
          me_cutflow_[key]->setBinLabel(4, Form("p_{T} > %.1f GeV", kMuonPtMinCuts_.at(station_id)));
          me_cutflow_[key]->setBinLabel(
              5, Form("%.2f < |#eta| < %.2f", kMuonEtaMinCuts_.at(station_id), kMuonEtaMaxCuts_.at(station_id)));

          me_cutflow_matched_[key] = bookNumerator1D(ibooker, me_cutflow_.at(key));
        }
      }  // shadowing
    }    // GEMChamber

    ////////////////////////////////////////////////////////////////////////////
    // Region - Station - iEta
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<const GEMEtaPartition*> eta_partition_vec = chamber_vec.front()->etaPartitions();
    if (not checkRefs(eta_partition_vec)) {
      edm::LogError(kLogCategory_) << "got an invalid ptr from GEMChamber::etaPartitions";
      continue;
    }

    for (const GEMEtaPartition* eta_partition : eta_partition_vec) {
      const int ieta = eta_partition->id().ieta();

      {  // shadowing
        const GEMDetId key = getReStEtKey(eta_partition->id());
        const TString gem_label = TString::Format("GE%d1-%c-E%d", station_id, (region_id > 0 ? 'P' : 'M'), ieta);
        const TString suffix = "_" + gem_label;
        const TString title = kMuonName_ + " " + gem_label;

        // FIXME
        const float dphi_up = (kMatchingMetric_ == MatchingMetric::kDeltaPhi) ? kMatchingCut_
                              : (kScenario_ == ScenarioOption::kCosmics)      ? 1.0
                                                                              : 0.2;
        me_residual_phi_[key] = ibooker.book1D("residual_phi" + suffix, title, 41, -dphi_up, dphi_up);
        me_residual_phi_[key]->setAxisTitle("Residual in global #phi [deg]", 1);

        if (kModeDev_) {
          // matching metric
          std::string matching_metric_x_title;
          if (kMatchingMetric_ == MatchingMetric::kDeltaPhi) {
            matching_metric_x_title = "#Delta#phi [deg]";

          } else if (kMatchingMetric_ == MatchingMetric::kRdPhi) {
            matching_metric_x_title = "R#Delta#phi [cm]";

          } else {
            matching_metric_x_title = "UNKNOWN METRIC";
          }

          // matching metrics without any cuts
          me_matching_metric_all_[key] =
              ibooker.book1D("matching_metric_all" + suffix, title, 101, -3 * kMatchingCut_, 3 * kMatchingCut_);
          me_matching_metric_all_[key]->setAxisTitle(matching_metric_x_title, 1);

          // matching metrics after cuts
          me_matching_metric_[key] =
              ibooker.book1D("matching_metric" + suffix, title, 101, -kMatchingCut_, kMatchingCut_);
          me_matching_metric_[key]->setAxisTitle(matching_metric_x_title, 1);

          // residuals in the global phi for muons (q < 0)
          me_residual_phi_muon_[key] =
              ibooker.book1D("residual_phi_muon" + suffix, title + " (#mu, q < 0)", 50, -0.5, 0.5);
          me_residual_phi_muon_[key]->setAxisTitle("Residual in global #phi [deg]", 1);
          me_residual_phi_muon_[key]->setAxisTitle("Number of muons", 2);

          // residuals in the global phi for anti-muons (q > 0)
          me_residual_phi_antimuon_[key] =
              ibooker.book1D("residual_phi_antimuon" + suffix, title + " (#tilde{#mu}, q > 0)", 50, -0.5, 0.5);
          me_residual_phi_antimuon_[key]->setAxisTitle("Residual in global #phi [deg]", 1);
          me_residual_phi_antimuon_[key]->setAxisTitle("Number of anti-muons", 2);

          // residuals in the local x
          me_residual_x_[key] = ibooker.book1D("residual_x" + suffix, title, 60, -1.5, 1.5);
          me_residual_x_[key]->setAxisTitle("Residual in local X [cm]", 1);

          // residuals in the local y
          me_residual_y_[key] = ibooker.book1D("residual_y" + suffix, title, 48, -12.0, 12.0);
          me_residual_y_[key]->setAxisTitle("Residual in local Y [cm]", 1);

          // the strip difference
          me_residual_strip_[key] = ibooker.book1D("residual_strip" + suffix, title, 21, -10.0, 10.0);
          me_residual_strip_[key]->setAxisTitle("propagation strip - hit strip", 1);
        }
      }  // shadowing
    }    // GEMEtaPartition
  }      // GEMStataion
}

// In the `cosmics` scenario, TODO doc
bool GEMEfficiencyAnalyzer::isInsideOut(const reco::Track& track) {
  return track.innerPosition().mag2() > track.outerPosition().mag2();
}

//
void GEMEfficiencyAnalyzer::buildGEMLayers(const GEMGeometry* gem) {
  std::map<GEMDetId, std::vector<const GEMChamber*> > chambers_per_layer;

  for (const GEMStation* station : gem->stations()) {
    const int region_id = station->region();
    const int station_id = station->station();
    const bool is_ge11 = station_id == 1;

    if (skipGEMStation(station_id)) {
      continue;
    }

    for (const GEMSuperChamber* superchamber : station->superChambers()) {
      // GE11: chamber == 0 for even chambers, chamber == 1 for odd chambers
      // GE21 and GE0: chamber == 0 for all chambers
      const int chamber_id = is_ge11 ? superchamber->id().chamber() % 2 : 0;

      for (const GEMChamber* chamber : superchamber->chambers()) {
        const int layer_id = chamber->id().layer();

        const GEMDetId key{region_id, 1, station_id, layer_id, chamber_id, 0};

        if (chambers_per_layer.find(key) == chambers_per_layer.end()) {
          chambers_per_layer.insert({key, std::vector<const GEMChamber*>()});
        }
        chambers_per_layer.at(key).push_back(chamber);
      }  // GEMChamber => iterate over layer ids
    }    // GEMSuperChamber => iterate over chamber ids
  }      // GEMStation

  gem_layers_.reserve(chambers_per_layer.size());
  for (auto [gem_id, chambers] : chambers_per_layer) {
    // layer position and rotation
    const float z_origin = chambers.front()->position().z();
    Surface::PositionType position{0.f, 0.f, z_origin};
    Surface::RotationType rotation;

    // eta partitions should have same R and Z spans.
    // XXX is it true?
    auto [r_min, r_max] = chambers.front()->surface().rSpan();
    auto [z_min, z_max] = chambers.front()->surface().zSpan();

    z_min -= z_origin;
    z_max -= z_origin;

    // the bounds from min and max R and Z in the local coordinates.
    SimpleDiskBounds* bounds = new SimpleDiskBounds(r_min, r_max, z_min, z_max);
    const Disk::DiskPointer layer = Disk::build(position, rotation, bounds);

    gem_layers_.emplace_back(layer, chambers, gem_id);

    LogDebug(kLogCategory_) << gem_id
                            << Form(" ==> (z_origin, z_min, z_max) = (%.2f, %.2f, %.2f)", z_origin, z_min, z_max);
  }  // ring
}

// TODO doc
// See https://twiki.cern.ch/twiki/pub/CMS/GEMPPDOfflineDQM/check-muon-direction.pdf
bool GEMEfficiencyAnalyzer::checkPropagationDirection(const reco::Track* track, const GEMLayer& layer) {
  const bool is_same_region = track->eta() * layer.id.region() > 0;

  bool skip = false;
  if (kScenario_ == ScenarioOption::kCosmics) {
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

GEMEfficiencyAnalyzer::StartingState GEMEfficiencyAnalyzer::buildStartingState(
    const reco::Muon& muon, const reco::TransientTrack& transient_track, const GEMLayer& gem_layer) {
  bool found = false;
  TrajectoryStateOnSurface state;
  DetId det_id;

  switch (kStartingStateType_) {
    case StartingStateType::kOutermostMeasurementState: {
      std::tie(found, state, det_id) = getOutermostMeasurementState(transient_track);
      break;
    }
    case StartingStateType::kInnermostMeasurementState: {
      std::tie(found, state, det_id) = getInnermostMeasurementState(transient_track);
      break;
    }
    case StartingStateType::kStateOnSurfaceWithCSCSegment: {
      std::tie(found, state, det_id) = buildStateOnSurfaceWithCSCSegment(muon, transient_track, gem_layer);
      break;
    }
    case StartingStateType::kAlignmentStyle: {
      std::tie(found, state, det_id) = buildStartingStateAlignmentStyle(muon, transient_track, gem_layer);
      break;
    }
    default: {
      edm::LogError(kLogCategory_) << "got an unexpected StartingStateType";
      break;
    }
  }

  found &= state.isValid();

  if (found and (det_id.det() == DetId::Detector::Muon)) {
    found &= isMuonSubdetAllowed(det_id, gem_layer.id.station());
  }

  if (found) {
    if (MuonHitHelper::isGEM(det_id)) {
      const GEMDetId start_id{det_id};

      const bool are_same_region = gem_layer.id.region() == start_id.region();
      const bool are_same_station = gem_layer.id.station() == start_id.station();
      const bool are_same_layer = gem_layer.id.layer() == start_id.layer();
      if (are_same_region and are_same_station and are_same_layer) {
        LogDebug(kLogCategory_)
            << "The starting detector of the muon propagation is same with the destination. Skip this propagation.";
        found = false;
      }
    }  // isGEM
  }    // found

  return std::make_tuple(found, state, det_id);
}

// Use the innermost measurement state as an initial state for the muon propagation.
// NOTE If the analyzer uses global or standalone muons and GEM hits are used in the
// muon reconstruction, the result should be biased.
// In 12_4_0_pre3, GEM hits are used in the pp scenario, but not in the cosmics scenario.
// https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/RecoMuon/StandAloneMuonProducer/python/standAloneMuons_cfi.py#L111-L127
// https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/RecoMuon/CosmicMuonProducer/python/cosmicMuons_cfi.py
GEMEfficiencyAnalyzer::StartingState GEMEfficiencyAnalyzer::getInnermostMeasurementState(
    const reco::TransientTrack& transient_track) {
  TrajectoryStateOnSurface state;
  DetId det_id;

  const reco::Track& track = transient_track.track();
  // get real innermost state
  if (isInsideOut(track)) {
    state = transient_track.outermostMeasurementState();
    det_id = track.outerDetId();

  } else {
    state = transient_track.innermostMeasurementState();
    det_id = track.innerDetId();
  }

  return std::make_tuple(true, state, det_id);
}

// Use the outermost measurement state as an initial state for the muon propagation.
GEMEfficiencyAnalyzer::StartingState GEMEfficiencyAnalyzer::getOutermostMeasurementState(
    const reco::TransientTrack& transient_track) {
  const reco::Track& track = transient_track.track();

  TrajectoryStateOnSurface state;
  DetId det_id;

  // get real innermost state
  if (isInsideOut(track)) {
    state = transient_track.innermostMeasurementState();
    det_id = track.innerDetId();

  } else {
    state = transient_track.outermostMeasurementState();
    det_id = track.outerDetId();
  }

  return std::make_tuple(true, state, det_id);
}

// Find the nearest CSC segment to the given GEM layer and then use a trajectory
// state on the surface with the segment as an initial state.
// XXX This method results in the residual phi distribution with two peaks
// because the muon and antimuon make different peaks.
GEMEfficiencyAnalyzer::StartingState GEMEfficiencyAnalyzer::buildStateOnSurfaceWithCSCSegment(
    const reco::Muon& muon, const reco::TransientTrack& transient_track, const GEMLayer& gem_layer) {
  bool found = false;
  TrajectoryStateOnSurface state;
  DetId det_id;

  if (const CSCSegment* csc_segment = findCSCSegment(muon, transient_track, gem_layer)) {
    const GeomDet* det = muon_service_->trackingGeometry()->idToDet(csc_segment->cscDetId());
    const GlobalPoint global_position = det->toGlobal(csc_segment->localPosition());

    found = true;
    state = transient_track.stateOnSurface(global_position);
    det_id = csc_segment->geographicalId();
  }

  return std::make_tuple(found, state, det_id);
}

// Find an ME11 segment and the build an initial state using the location and
// direction of the ME11 segment. If the muon has an inner track, the outerP of
// the inner track is used as the momentum magnitude. If not, the momentum
// magnitude is set to 1 GeV.
// https://github.com/gem-sw/alignment/blob/713e8fa/GEMCSCBendingAnalyzer/MuonAnalyser/plugins/analyser.cc#L435-L446
GEMEfficiencyAnalyzer::StartingState GEMEfficiencyAnalyzer::buildStartingStateAlignmentStyle(
    const reco::Muon& muon, const reco::TransientTrack& transient_track, const GEMLayer& gem_layer) {
  bool found = false;
  TrajectoryStateOnSurface state;
  DetId det_id;

  if (const CSCSegment* csc_segment = findCSCSegment(muon, transient_track, gem_layer)) {
    found = true;
    det_id = csc_segment->geographicalId();

    // position
    const LocalPoint position = csc_segment->localPosition();
    // momentum
    const reco::TrackRef inner_track = muon.innerTrack();
    const float momentum_magnitude = inner_track.isNonnull() ? inner_track.get()->outerP() : 1.0f;
    const LocalVector momentum = momentum_magnitude * csc_segment->localDirection();

    // trajectory parameter
    const LocalTrajectoryParameters trajectory_parameters{position, momentum, muon.charge()};

    // trajectory error
    const LocalTrajectoryError trajectory_error =
        asSMatrix<5>(csc_segment->parametersError().similarityT(csc_segment->projectionMatrix()));

    // surface
    const Plane& surface = muon_service_->trackingGeometry()->idToDet(det_id)->surface();

    state =
        TrajectoryStateOnSurface{trajectory_parameters, trajectory_error, surface, &*muon_service_->magneticField()};
  }

  return std::make_tuple(found, state, det_id);
}

//
// for beam scenario
const CSCSegment* GEMEfficiencyAnalyzer::findCSCSegmentBeam(const reco::TransientTrack& transient_track,
                                                            const GEMLayer& gem_layer) {
  const CSCSegment* best_csc_segment = nullptr;
  double min_z_distance = std::numeric_limits<double>::infinity();  // in cm

  for (trackingRecHit_iterator tracking_rechit_iter = transient_track.recHitsBegin();
       tracking_rechit_iter != transient_track.recHitsEnd();
       tracking_rechit_iter++) {
    const TrackingRecHit* tracking_rechit = *tracking_rechit_iter;
    if (not tracking_rechit->isValid()) {
      LogDebug(kLogCategory_) << "got an invalid trackingRecHit_iterator from transient_track. skip it.";
      continue;
    }

    const DetId det_id = tracking_rechit->geographicalId();
    if (not MuonHitHelper::isCSC(det_id)) {
      continue;
    }

    if (tracking_rechit->dimension() != kCSCSegmentDimension_) {
      continue;
    }

    const CSCDetId csc_id{det_id};
    if (not isCSCAllowed(csc_id, gem_layer.id.station())) {
      continue;
    }

    if (auto csc_segment = dynamic_cast<const CSCSegment*>(tracking_rechit)) {
      const GeomDet* det = muon_service_->trackingGeometry()->idToDet(csc_id);
      if (det == nullptr) {
        edm::LogError(kLogCategory_) << "GlobalTrackingGeometry::idToDet returns nullptr; CSCDetId=" << csc_id;
        continue;
      }
      const GlobalPoint global_position = det->toGlobal(csc_segment->localPosition());
      const float z_distance = std::abs(gem_layer.disk->localZclamped(global_position));

      if (z_distance < min_z_distance) {
        best_csc_segment = csc_segment;
        min_z_distance = z_distance;
      }

    } else {
      edm::LogError(kLogCategory_)
          << "failed to perform the conversion from `const TrackingRechit*` to `const CSCSegment*`";
    }
  }  // trackingRecHit_iterator

  return best_csc_segment;
}

const CSCSegment* GEMEfficiencyAnalyzer::findCSCSegmentCosmics(const reco::Muon& muon, const GEMLayer& gem_layer) {
  const CSCSegment* best_csc_segment = nullptr;

  for (const reco::MuonChamberMatch& chamber_match : muon.matches()) {
    if (not MuonHitHelper::isCSC(chamber_match.id)) {
      continue;
    }

    const CSCDetId csc_id{chamber_match.id};
    if (not isCSCAllowed(csc_id, gem_layer.id.station())) {
      continue;
    }

    const float x_track = chamber_match.x;
    const float y_track = chamber_match.y;

    for (const reco::MuonSegmentMatch& segment_match : chamber_match.segmentMatches) {
      if (not segment_match.isMask(reco::MuonSegmentMatch::BestInStationByDR)) {
        continue;
      }

      const float dr = std::hypot(x_track - segment_match.x, y_track - segment_match.y);
      std::cout << kLogCategory_ << ": dr=" << dr << std::endl;

      if (dr > kMuonSegmentMatchDRCut_) {
        LogDebug(kLogCategory_) << "too large dR(muon, segment)";
        break;
      }

      if (segment_match.cscSegmentRef.isNonnull()) {
        best_csc_segment = segment_match.cscSegmentRef.get();
      }
    }  // MuonSegmentMatch
  }    // MuonChamberMatch

  return best_csc_segment;
}

// just thin wrapper
const CSCSegment* GEMEfficiencyAnalyzer::findCSCSegment(const reco::Muon& muon,
                                                        const reco::TransientTrack& transient_track,
                                                        const GEMLayer& gem_layer) {
  if (kScenario_ == ScenarioOption::kCosmics) {
    return findCSCSegmentCosmics(muon, gem_layer);
  } else {
    // pp or HI
    return findCSCSegmentBeam(transient_track, gem_layer);
  }
}

bool GEMEfficiencyAnalyzer::isMuonSubdetAllowed(const DetId& det_id, const int gem_station) {
  if ((gem_station < 0) or (gem_station > 2)) {
    edm::LogError(kLogCategory_) << "got unexpected gem station " << gem_station;
    return false;
  }

  if (det_id.det() != DetId::Detector::Muon) {
    edm::LogError(kLogCategory_) << Form(
        "(Detector, Subdetector) = (%d, %d)", static_cast<int>(det_id.det()), det_id.subdetId());
    return false;
  }

  const std::vector<int> allowed = kMuonSubdetForGEM_.at(gem_station);
  return allowed.empty() or (std::find(allowed.begin(), allowed.end(), det_id.subdetId()) != allowed.end());
}

// Returns a bool value indicating whether or not the CSC detector can be used
// as a start detector for a given GEM station.
// See https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/DataFormats/MuonDetId/interface/CSCDetId.h#L187-L193
// This method is used when using `buildStateOnSurfaceWithCSCSegment` or
// `buildStartingStateAlignmentStyle`
bool GEMEfficiencyAnalyzer::isCSCAllowed(const CSCDetId& csc_id, const int gem_station) {
  if ((gem_station < 0) or (gem_station > 2)) {
    edm::LogError(kLogCategory_) << "got unexpected gem station " << gem_station;
    return false;
  }

  // unsigned short to int
  const int csc_chamber_type = static_cast<int>(csc_id.iChamberType());

  const std::vector<int> allowed = kCSCForGEM_.at(gem_station);
  return allowed.empty() or (std::find(allowed.begin(), allowed.end(), csc_chamber_type) != allowed.end());
}

bool GEMEfficiencyAnalyzer::checkBounds(const Plane& plane, const GlobalPoint& global_point) {
  const LocalPoint local_point = plane.toLocal(global_point);
  const LocalPoint local_point_2d(local_point.x(), local_point.y(), 0.0f);
  return plane.bounds().inside(local_point_2d);
}

// TODO comment on the scale
// https://github.com/cms-sw/cmssw/blob/CMSSW_12_0_0_pre3/DataFormats/GeometrySurface/src/SimpleDiskBounds.cc#L20-L35
bool GEMEfficiencyAnalyzer::checkBounds(const Plane& plane,
                                        const GlobalPoint& global_point,
                                        const GlobalError& global_error,
                                        const float scale) {
  const LocalPoint local_point = plane.toLocal(global_point);
  const LocalError local_error = ErrorFrameTransformer::transform(global_error, plane);

  const LocalPoint local_point_2d{local_point.x(), local_point.y(), 0.0f};
  return plane.bounds().inside(local_point_2d, local_error, scale);
}

const GEMEtaPartition* GEMEfficiencyAnalyzer::findEtaPartition(const GlobalPoint& global_point,
                                                               const GlobalError& global_error,
                                                               const std::vector<const GEMChamber*>& chamber_vector) {
  const GEMEtaPartition* bound = nullptr;
  for (const GEMChamber* chamber : chamber_vector) {
    if (not checkBounds(chamber->surface(), global_point, global_error, kBoundsErrorScale_)) {
      continue;
    }

    for (const GEMEtaPartition* eta_partition : chamber->etaPartitions()) {
      if (checkBounds(eta_partition->surface(), global_point, global_error, kBoundsErrorScale_)) {
        bound = eta_partition;
        break;
      }
    }  // GEMEtaPartition
  }    // GEMChamber

  return bound;
}

// Borrowed from https://github.com/gem-sw/alignment/blob/713e8fa/GEMCSCBendingAnalyzer/MuonAnalyser/plugins/analyser.cc#L321-L327
float GEMEfficiencyAnalyzer::computeRdPhi(const GlobalPoint& prop_global_pos,
                                          const LocalPoint& hit_local_pos,
                                          const GEMEtaPartition* eta_partition) {
  const StripTopology& topology = eta_partition->specificTopology();
  const LocalPoint prop_local_pos = eta_partition->toLocal(prop_global_pos);

  const float dx = prop_local_pos.x() - hit_local_pos.x();
  const float dy = prop_local_pos.y() - hit_local_pos.y();
  const float hit_strip = eta_partition->strip(hit_local_pos);
  const float hit_phi = topology.stripAngle(hit_strip);
  const float rdphi = std::cos(hit_phi) * dx - std::sin(hit_phi) * dy;
  return rdphi;
}

// Returns a global delta phi between a propagated muon and a reconstructed hit.
float GEMEfficiencyAnalyzer::computeDeltaPhi(const GlobalPoint& prop_global_pos,
                                             const LocalPoint& hit_local_pos,
                                             const GEMEtaPartition* eta_partition) {
  const GlobalPoint hit_global_pos = eta_partition->toGlobal(hit_local_pos);
  const float dphi = Geom::convertRadToDeg(prop_global_pos.phi() - hit_global_pos.phi());
  return dphi;
}

// a thin wrapper to hide a messy conditional statement
float GEMEfficiencyAnalyzer::computeMatchingMetric(const GlobalPoint& prop_global_pos,
                                                   const LocalPoint& hit_local_pos,
                                                   const GEMEtaPartition* eta_partition) {
  float metric;
  switch (kMatchingMetric_) {
    case MatchingMetric::kDeltaPhi: {
      metric = computeDeltaPhi(prop_global_pos, hit_local_pos, eta_partition);
      break;
    }
    case MatchingMetric::kRdPhi: {
      metric = computeRdPhi(prop_global_pos, hit_local_pos, eta_partition);
      break;
    }
    default: {
      edm::LogError(kLogCategory_) << "unknown MatchingMetric.";  // TODO
      metric = std::numeric_limits<float>::infinity();
    }
  }

  return metric;
}

// This method finds the closest hit to a propagated muon in the eta partition
// with that propagated muon. Adjacent eta partitions are excluded from the area
// of interst to avoid ambiguity in defining the detection efficiency of each
// eta partition.
std::pair<const GEMRecHit*, float> GEMEfficiencyAnalyzer::findClosestHit(const GlobalPoint& prop_global_pos,
                                                                         const GEMRecHitCollection::range& rechit_range,
                                                                         const GEMEtaPartition* eta_partition) {
  const GEMRecHit* closest_hit = nullptr;
  float min_metric = std::numeric_limits<float>::infinity();

  for (auto hit = rechit_range.first; hit != rechit_range.second; ++hit) {
    const LocalPoint hit_local_pos = hit->localPosition();

    const float metric = computeMatchingMetric(prop_global_pos, hit_local_pos, eta_partition);

    if (std::abs(metric) < std::abs(min_metric)) {
      min_metric = metric;
      closest_hit = &(*hit);
    }
  }

  return std::make_pair(closest_hit, min_metric);
}

void GEMEfficiencyAnalyzer::dqmBeginRun(edm::Run const&, edm::EventSetup const& setup) {
  const GEMGeometry* gem = nullptr;
  if (auto handle = setup.getHandle(kGEMGeometryTokenBeginRun_)) {
    gem = handle.product();
  } else {
    edm::LogError(kLogCategory_ + "|dqmBeginRun") << "failed to get GEMGeometry";
    return;
  }

  buildGEMLayers(gem);
}

void GEMEfficiencyAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  //////////////////////////////////////////////////////////////////////////////
  // get data from Event
  //////////////////////////////////////////////////////////////////////////////
  const GEMRecHitCollection* rechit_collection = nullptr;
  if (auto handle = event.getHandle(kGEMRecHitCollectionToken_)) {
    rechit_collection = handle.product();
  } else {
    edm::LogError(kLogCategory_) << "failed to get GEMRecHitCollection";
    return;
  }

  const edm::View<reco::Muon>* muon_view = nullptr;
  if (auto handle = event.getHandle(kMuonViewToken_)) {
    muon_view = handle.product();
  } else {
    edm::LogError(kLogCategory_) << "failed to get View<Muon>";
    return;
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
  // get data from EventSetup
  //////////////////////////////////////////////////////////////////////////////
  const TransientTrackBuilder* transient_track_builder = nullptr;
  if (auto handle = setup.getHandle(kTransientTrackBuilderToken_)) {
    transient_track_builder = handle.product();
  } else {
    edm::LogError(kLogCategory_) << "failed to get TransientTrackBuilder";
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // get more data from EventSetup using MuonServiceProxy
  //////////////////////////////////////////////////////////////////////////////
  muon_service_->update(setup);

  // TODO StraightLinePropagator if B < epsilon else SteppingHelixPropagatorAny
  const Propagator* propagator = nullptr;
  if (auto handle = muon_service_->propagator("SteppingHelixPropagatorAny")) {
    propagator = handle.product();
  } else {
    edm::LogError(kLogCategory_) << "failed to get Propagator";
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //  Main loop
  //////////////////////////////////////////////////////////////////////////////

  for (const reco::Muon& muon : *muon_view) {
    const reco::Track* track = muon.muonTrack(kMuonTrackType_).get();
    if (track == nullptr) {
      LogDebug(kLogCategory_) << "failed to get a " << kMuonTrackTypeName_;
      continue;
    }

    const reco::TransientTrack transient_track = transient_track_builder->build(track);
    if (not transient_track.isValid()) {
      edm::LogError(kLogCategory_) << "failed to build TransientTrack";
      continue;
    }

    for (const GEMLayer& layer : gem_layers_) {
      if (checkPropagationDirection(track, layer)) {
        LogDebug(kLogCategory_) << "bad flight path. skip this propagation.";
        continue;
      }

      const auto [found_start_state, start_state, start_id] = buildStartingState(muon, transient_track, layer);
      if (not found_start_state) {
        LogDebug(kLogCategory_) << "propagation starting state not found";
        continue;
      }

      // the trajectory state on the destination surface
      const auto [propagated_state, prop_path_length] = propagator->propagateWithPath(start_state, *(layer.disk));
      if (not propagated_state.isValid()) {
        LogDebug(kLogCategory_) << "failed to propagate a muon from "
                                << Form("(Detector, Subdetector) = (%d, %d)",
                                        static_cast<int>(start_id.det()),
                                        start_id.subdetId())
                                << " to " << layer.id << ". The path length is " << prop_path_length;
        continue;
      }

      const GlobalPoint prop_global_pos = propagated_state.globalPosition();
      const GlobalError& prop_global_err =
          ErrorFrameTransformer::transform(propagated_state.localError().positionError(), *layer.disk);

      if (not checkBounds(*layer.disk, prop_global_pos, prop_global_err, kBoundsErrorScale_)) {
        LogDebug(kLogCategory_) << "failed to pass checkBounds";
        continue;
      }

      const GEMEtaPartition* eta_partition = findEtaPartition(prop_global_pos, prop_global_err, layer.chambers);
      if (eta_partition == nullptr) {
        LogDebug(kLogCategory_) << "failed to find an eta partition";
        continue;
      }

      const GEMDetId gem_id = eta_partition->id();

      if (kMaskChamberWithError_) {
        const bool has_error = maskChamberWithError(gem_id.chamberId(), oh_status_collection, vfat_status_collection);
        if (has_error) {
          LogDebug(kLogCategory_) << gem_id.chamberId() << " has an erorr. Skip this propagation.";
          continue;
        }
      }

      //////////////////////////////////////////////////////////////////////////
      //
      //////////////////////////////////////////////////////////////////////////
      const GEMDetId rs_key = getReStKey(gem_id);     // region-station
      const GEMDetId rsl_key = getReStLaKey(gem_id);  // region-station-layer
      const GEMDetId rse_key = getReStEtKey(gem_id);  // region-station-ieta

      const int station_id = gem_id.station();
      const int chamber_id = gem_id.chamber();
      const int ieta = gem_id.ieta();

      const double muon_pt = muon.pt();
      const double muon_eta = std::fabs(muon.eta());
      const double muon_phi = Geom::convertRadToDeg(muon.phi());

      const double prop_global_err_r = std::sqrt(prop_global_err.rerr(prop_global_pos));
      const double prop_global_err_phi = Geom::convertRadToDeg(std::sqrt(prop_global_err.phierr(prop_global_pos)));

      // cuts
      const bool passed_prop_err_r_cut = (prop_global_err_r < kPropagationErrorRCut_);
      const bool passed_prop_err_phi_cut = (prop_global_err_phi < kPropagationErrorPhiCut_);
      const bool passed_pt_cut = muon_pt > kMuonPtMinCuts_.at(station_id);
      const bool passed_eta_cut =
          (muon_eta > kMuonEtaMinCuts_.at(station_id)) and (muon_eta < kMuonEtaMaxCuts_.at(station_id));

      const bool passed_prop_err_cuts = passed_prop_err_r_cut and passed_prop_err_phi_cut;
      const bool passed_all_cuts = passed_prop_err_cuts and passed_pt_cut and passed_eta_cut;

      const int cutflow_last = not kModeDev_                 ? 0
                               : not passed_prop_err_r_cut   ? 1
                               : not passed_prop_err_phi_cut ? 2
                               : not passed_pt_cut           ? 3
                               : not passed_eta_cut          ? 4
                                                             : 5;

      //////////////////////////////////////////////////////////////////////////
      // Fill denominators
      //////////////////////////////////////////////////////////////////////////
      if (passed_eta_cut and passed_prop_err_cuts) {
        fillMEWithinLimits(me_muon_pt_, rs_key, muon_pt);
      }

      if (passed_pt_cut and passed_prop_err_cuts) {
        fillMEWithinLimits(me_muon_eta_, rs_key, muon_eta);
      }

      if (passed_all_cuts) {
        fillME(me_chamber_ieta_, rsl_key, chamber_id, ieta);
        fillME(me_muon_phi_, rs_key, muon_phi);
      }

      if (kModeDev_) {
        fillMEWithinLimits(me_prop_path_length_, rsl_key, prop_path_length);
        fillMEWithinLimits(me_prop_err_r_, rsl_key, prop_global_err_r);
        fillMEWithinLimits(me_prop_err_phi_, rsl_key, prop_global_err_phi);

        fillMEWithinLimits(me_muon_pt_all_, rs_key, muon_pt);
        fillMEWithinLimits(me_muon_eta_all_, rs_key, muon_eta);

        fillME(me_muon_charge_, rs_key, muon.charge());

        for (int bin = 1; bin <= cutflow_last; bin++) {
          fillME(me_cutflow_, rsl_key, bin);
        }

      }  // dev mode

      //////////////////////////////////////////////////////////////////////////
      // Find a closet hit
      //////////////////////////////////////////////////////////////////////////
      const auto [hit, matching_metric] =
          findClosestHit(prop_global_pos, rechit_collection->get(gem_id), eta_partition);

      if (hit == nullptr) {
        LogDebug(kLogCategory_) << "hit not found";
        continue;
      }

      if (kModeDev_) {
        fillMEWithinLimits(me_matching_metric_all_, rse_key, matching_metric);
      }

      if (std::abs(matching_metric) > kMatchingCut_) {
        LogDebug(kLogCategory_) << "failed to pass the residual rphi cut";
        continue;
      }

      //////////////////////////////////////////////////////////////////////////
      // Fill numerators
      //////////////////////////////////////////////////////////////////////////
      if (passed_eta_cut and passed_prop_err_cuts) {
        fillMEWithinLimits(me_muon_pt_matched_, rs_key, muon_pt);
      }

      if (passed_pt_cut and passed_prop_err_cuts) {
        fillMEWithinLimits(me_muon_eta_matched_, rs_key, muon_eta);
      }

      if (passed_all_cuts) {
        fillME(me_chamber_ieta_matched_, rsl_key, chamber_id, ieta);
        fillME(me_muon_phi_matched_, rs_key, muon_phi);
      }

      if (kModeDev_) {
        fillMEWithinLimits(me_prop_path_length_matched_, rsl_key, prop_path_length);

        fillMEWithinLimits(me_prop_err_r_matched_, rsl_key, prop_global_err_r);
        fillMEWithinLimits(me_prop_err_phi_matched_, rsl_key, prop_global_err_phi);

        fillMEWithinLimits(me_muon_pt_all_matched_, rs_key, muon_pt);
        fillMEWithinLimits(me_muon_eta_all_matched_, rs_key, muon_eta);

        fillME(me_muon_charge_matched_, rs_key, muon.charge());

        if (passed_all_cuts) {
          for (int bin = 1; bin <= cutflow_last; bin++) {
            fillME(me_cutflow_matched_, rsl_key, bin);
          }
        }
      }

      //////////////////////////////////////////////////////////////////////////
      // Fill resolutions
      //////////////////////////////////////////////////////////////////////////
      if (passed_all_cuts) {
        const LocalPoint hit_local_pos = hit->localPosition();
        const GlobalPoint& hit_global_pos = eta_partition->toGlobal(hit_local_pos);
        const float residual_phi = Geom::convertRadToDeg(prop_global_pos.phi() - hit_global_pos.phi());

        fillMEWithinLimits(me_residual_phi_, rse_key, residual_phi);

        if (kModeDev_) {
          const LocalPoint prop_local_pos = eta_partition->toLocal(prop_global_pos);
          const StripTopology& topology = eta_partition->specificTopology();

          const float residual_x = prop_local_pos.x() - hit_local_pos.x();
          const float residual_y = prop_local_pos.y() - hit_local_pos.y();
          const float residual_strip = topology.strip(prop_local_pos) - topology.strip(hit_local_pos);

          fillMEWithinLimits(me_matching_metric_, rse_key, matching_metric);
          fillMEWithinLimits(me_residual_x_, rse_key, residual_x);
          fillMEWithinLimits(me_residual_y_, rse_key, residual_y);
          fillMEWithinLimits(me_residual_strip_, rse_key, residual_strip);

          if (muon.charge() < 0) {
            fillMEWithinLimits(me_residual_phi_muon_, rse_key, residual_phi);
          } else {
            fillMEWithinLimits(me_residual_phi_antimuon_, rse_key, residual_phi);
          }
        }  // kModeDev_
      }    // passed_all_cuts
    }      // destination
  }        // Muon
}  // analyze
