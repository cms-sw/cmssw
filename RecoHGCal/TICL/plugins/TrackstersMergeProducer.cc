#include <memory>  // unique_ptr
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "RecoHGCal/TICL/plugins/LinkingAlgoBase.h"
#include "RecoHGCal/TICL/plugins/LinkingAlgoFactory.h"
#include "RecoHGCal/TICL/plugins/LinkingAlgoByDirectionGeometric.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackstersPCA.h"

using namespace ticl;
using cms::Ort::ONNXRuntime;

class TrackstersMergeProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit TrackstersMergeProducer(const edm::ParameterSet &ps, const ONNXRuntime *);
  ~TrackstersMergeProducer() override {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &iConfig);
  static void globalEndJob(const ONNXRuntime *);

  void beginJob();
  void endJob();

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;

private:
  typedef ticl::Trackster::IterationIndex TracksterIterIndex;
  typedef ticl::Vector Vector;

  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, TracksterIterIndex);

  void printTrackstersDebug(const std::vector<Trackster> &, const char *label) const;
  void assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates) const;
  void dumpTrackster(const Trackster &) const;

  std::unique_ptr<LinkingAlgoBase> linkingAlgo_;

  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_clue3d_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_quality_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_err_token_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const bool regressionAndPid_;
  std::unique_ptr<TracksterInferenceAlgoBase> inferenceAlgo_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const std::string detector_;
  const std::string propName_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const bool optimiseAcrossTracksters_;
  const bool useMTDTiming_;
  const int eta_bin_window_;
  const int phi_bin_window_;
  const double pt_sigma_high_;
  const double pt_sigma_low_;
  const double halo_max_distance2_;
  const double track_min_pt_;
  const double track_min_eta_;
  const double track_max_eta_;
  const int track_max_missing_outerhits_;
  const double cosangle_align_;
  const double e_over_h_threshold_;
  const double pt_neutral_threshold_;
  const double resol_calo_offset_had_;
  const double resol_calo_scale_had_;
  const double resol_calo_offset_em_;
  const double resol_calo_scale_em_;
  std::once_flag initializeGeometry_;

  const HGCalDDDConstants *hgcons_;

  std::unique_ptr<GeomDet> firstDisk_[2];

  hgcal::RecHitTools rhtools_;

  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
};

TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps, const ONNXRuntime *)
    : tracksters_clue3d_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersclue3d"))),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      regressionAndPid_(ps.getParameter<bool>("regressionAndPid")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      optimiseAcrossTracksters_(ps.getParameter<bool>("optimiseAcrossTracksters")),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      eta_bin_window_(ps.getParameter<int>("eta_bin_window")),
      phi_bin_window_(ps.getParameter<int>("phi_bin_window")),
      pt_sigma_high_(ps.getParameter<double>("pt_sigma_high")),
      pt_sigma_low_(ps.getParameter<double>("pt_sigma_low")),
      halo_max_distance2_(ps.getParameter<double>("halo_max_distance2")),
      track_min_pt_(ps.getParameter<double>("track_min_pt")),
      track_min_eta_(ps.getParameter<double>("track_min_eta")),
      track_max_eta_(ps.getParameter<double>("track_max_eta")),
      track_max_missing_outerhits_(ps.getParameter<int>("track_max_missing_outerhits")),
      cosangle_align_(ps.getParameter<double>("cosangle_align")),
      e_over_h_threshold_(ps.getParameter<double>("e_over_h_threshold")),
      pt_neutral_threshold_(ps.getParameter<double>("pt_neutral_threshold")),
      resol_calo_offset_had_(ps.getParameter<double>("resol_calo_offset_had")),
      resol_calo_scale_had_(ps.getParameter<double>("resol_calo_scale_had")),
      resol_calo_offset_em_(ps.getParameter<double>("resol_calo_offset_em")),
      resol_calo_scale_em_(ps.getParameter<double>("resol_calo_scale_em")) {
  produces<std::vector<Trackster>>();
  produces<std::vector<TICLCandidate>>();

  if (useMTDTiming_) {
    tracks_time_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTime"));
    tracks_time_quality_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeQual"));
    tracks_time_err_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeErr"));
  }

  std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  hdc_token_ =
      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));

  auto linkingPSet = ps.getParameter<edm::ParameterSet>("linkingPSet");
  auto algoType = linkingPSet.getParameter<std::string>("type");
  linkingAlgo_ = LinkingAlgoFactory::get()->create(algoType, linkingPSet);

  std::string inferencePlugin = ps.getParameter<std::string>("inferenceAlgo");
  edm::ParameterSet inferencePSet = ps.getParameter<edm::ParameterSet>("pluginInferenceAlgo" + inferencePlugin);
  inferenceAlgo_ = std::unique_ptr<TracksterInferenceAlgoBase>(
      TracksterInferenceAlgoFactory::get()->create(inferencePlugin, inferencePSet));
}

void TrackstersMergeProducer::beginJob() {}

void TrackstersMergeProducer::endJob() {}

std::unique_ptr<ONNXRuntime> TrackstersMergeProducer::initializeGlobalCache(const edm::ParameterSet &iConfig) {
  return std::unique_ptr<ONNXRuntime>(nullptr);
}

void TrackstersMergeProducer::globalEndJob(const ONNXRuntime *) {}

void TrackstersMergeProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  hgcons_ = hdc.product();

  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);

  edm::ESHandle<MagneticField> bfield = es.getHandle(bfield_token_);
  edm::ESHandle<Propagator> propagator = es.getHandle(propagator_token_);

  linkingAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
};

void TrackstersMergeProducer::fillTile(TICLTracksterTiles &tracksterTile,
                                       const std::vector<Trackster> &tracksters,
                                       TracksterIterIndex tracksterIteration) {
  int tracksterId = 0;
  for (auto const &t : tracksters) {
    tracksterTile.fill(tracksterIteration, t.barycenter().eta(), t.barycenter().phi(), tracksterId);
    LogDebug("TrackstersMergeProducer") << "Adding tracksterId: " << tracksterId << " into bin [eta,phi]: [ "
                                        << tracksterTile[tracksterIteration].etaBin(t.barycenter().eta()) << ", "
                                        << tracksterTile[tracksterIteration].phiBin(t.barycenter().phi())
                                        << "] for iteration: " << tracksterIteration << std::endl;

    tracksterId++;
  }
}

void TrackstersMergeProducer::dumpTrackster(const Trackster &t) const {
  auto e_over_h = (t.raw_em_pt() / ((t.raw_pt() - t.raw_em_pt()) != 0. ? (t.raw_pt() - t.raw_em_pt()) : 1.));
  LogDebug("TrackstersMergeProducer")
      << "\nTrackster raw_pt: " << t.raw_pt() << " raw_em_pt: " << t.raw_em_pt() << " eoh: " << e_over_h
      << " barycenter: " << t.barycenter() << " eta,phi (baricenter): " << t.barycenter().eta() << ", "
      << t.barycenter().phi() << " eta,phi (eigen): " << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi()
      << " pt(eigen): " << std::sqrt(t.eigenvectors(0).Unit().perp2()) * t.raw_energy() << " seedID: " << t.seedID()
      << " seedIndex: " << t.seedIndex() << " size: " << t.vertices().size() << " average usage: "
      << (std::accumulate(std::begin(t.vertex_multiplicity()), std::end(t.vertex_multiplicity()), 0.) /
          (float)t.vertex_multiplicity().size())
      << " raw_energy: " << t.raw_energy() << " regressed energy: " << t.regressed_energy()
      << " probs(ga/e/mu/np/cp/nh/am/unk): ";
  for (auto const &p : t.id_probabilities()) {
    LogDebug("TrackstersMergeProducer") << std::fixed << p << " ";
  }
  LogDebug("TrackstersMergeProducer") << " sigmas: ";
  for (auto const &s : t.sigmas()) {
    LogDebug("TrackstersMergeProducer") << s << " ";
  }
  LogDebug("TrackstersMergeProducer") << std::endl;
}

void TrackstersMergeProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  auto resultTrackstersMerged = std::make_unique<std::vector<Trackster>>();
  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();
  auto resultFromTracks = std::make_unique<std::vector<TICLCandidate>>();

  edm::Handle<std::vector<Trackster>> trackstersclue3d_h;
  evt.getByToken(tracksters_clue3d_token_, trackstersclue3d_h);

  edm::Handle<std::vector<reco::Track>> track_h;
  evt.getByToken(tracks_token_, track_h);
  const auto &tracks = *track_h;

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);
  const auto &muons = evt.get(muons_token_);
  edm::Handle<edm::ValueMap<float>> trackTime_h;
  edm::Handle<edm::ValueMap<float>> trackTimeErr_h;
  edm::Handle<edm::ValueMap<float>> trackTimeQual_h;
  if (useMTDTiming_) {
    evt.getByToken(tracks_time_token_, trackTime_h);
    evt.getByToken(tracks_time_err_token_, trackTimeErr_h);
    evt.getByToken(tracks_time_quality_token_, trackTimeQual_h);
  }

  // Linking
  linkingAlgo_->linkTracksters(track_h,
                               trackTime_h,
                               trackTimeErr_h,
                               trackTimeQual_h,
                               muons,
                               trackstersclue3d_h,
                               useMTDTiming_,
                               *resultCandidates,
                               *resultFromTracks);

  // Print debug info
  LogDebug("TrackstersMergeProducer") << "Results from the linking step : " << std::endl
                                      << "No. of Tracks : " << tracks.size()
                                      << "  No. of Tracksters : " << (*trackstersclue3d_h).size() << std::endl
                                      << "(neutral candidates have track id -1)" << std::endl;

  std::vector<TICLCandidate> &candidates = *resultCandidates;
  for (const auto &cand : candidates) {
    auto track_ptr = cand.trackPtr();
    auto trackster_ptrs = cand.tracksters();

    auto track_idx = track_ptr.get() - (edm::Ptr<reco::Track>(track_h, 0)).get();
    track_idx = (track_ptr.isNull()) ? -1 : track_idx;
#ifdef EDM_ML_DEBUG
    LogDebug("TrackstersMergeProducer") << "PDG ID " << cand.pdgId() << " charge " << cand.charge() << " p " << cand.p()
                                        << std::endl;
    LogDebug("TrackstersMergeProducer") << "track id (p) : " << track_idx << " ("
                                        << (track_ptr.isNull() ? -1 : track_ptr->p()) << ") "
                                        << " trackster ids (E) : ";
#endif

    // Merge included tracksters
    ticl::Trackster outTrackster;
    outTrackster.addTrackIdx(track_idx);
    auto updated_size = 0;
    for (const auto &ts_ptr : trackster_ptrs) {
#ifdef EDM_ML_DEBUG
      auto ts_idx = ts_ptr.get() - (edm::Ptr<ticl::Trackster>(trackstersclue3d_h, 0)).get();
      LogDebug("TrackstersMergeProducer") << ts_idx << " (" << ts_ptr->raw_energy() << ") ";
#endif

      auto &thisTrackster = *ts_ptr;
      updated_size += thisTrackster.vertices().size();
      outTrackster.vertices().reserve(updated_size);
      outTrackster.vertex_multiplicity().reserve(updated_size);
      std::copy(std::begin(thisTrackster.vertices()),
                std::end(thisTrackster.vertices()),
                std::back_inserter(outTrackster.vertices()));
      std::copy(std::begin(thisTrackster.vertex_multiplicity()),
                std::end(thisTrackster.vertex_multiplicity()),
                std::back_inserter(outTrackster.vertex_multiplicity()));
    }

    LogDebug("TrackstersMergeProducer") << std::endl;

    // Find duplicate LCs
    auto &orig_vtx = outTrackster.vertices();
    auto vtx_sorted{orig_vtx};
    std::sort(std::begin(vtx_sorted), std::end(vtx_sorted));
    for (unsigned int iLC = 1; iLC < vtx_sorted.size(); ++iLC) {
      if (vtx_sorted[iLC] == vtx_sorted[iLC - 1]) {
        // Clean up duplicate LCs
        const auto lcIdx = vtx_sorted[iLC];
        const auto firstEl = std::find(orig_vtx.begin(), orig_vtx.end(), lcIdx);
        const auto firstPos = std::distance(std::begin(orig_vtx), firstEl);
        auto iDup = std::find(std::next(firstEl), orig_vtx.end(), lcIdx);
        while (iDup != orig_vtx.end()) {
          orig_vtx.erase(iDup);
          outTrackster.vertex_multiplicity().erase(outTrackster.vertex_multiplicity().begin() +
                                                   std::distance(std::begin(orig_vtx), iDup));
          outTrackster.vertex_multiplicity()[firstPos] -= 1;
          iDup = std::find(std::next(firstEl), orig_vtx.end(), lcIdx);
        };
      }
    }

    outTrackster.zeroProbabilities();
    if (!outTrackster.vertices().empty()) {
      resultTrackstersMerged->push_back(outTrackster);
    }
  }

  assignPCAtoTracksters(*resultTrackstersMerged,
                        layerClusters,
                        layerClustersTimes,
                        rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                        rhtools_);
  inferenceAlgo_->inputData(layerClusters, *resultTrackstersMerged, rhtools_);
  inferenceAlgo_->runInference(
      *resultTrackstersMerged);  //option to use "Linking" instead of "CLU3D"/"energyAndPid" instead of "PID"

  //filling the TICLCandidates information
  assert(resultTrackstersMerged->size() == resultCandidates->size());

  auto isHad = [](const Trackster &tracksterMerge) {
    return tracksterMerge.id_probability(Trackster::ParticleType::photon) +
               tracksterMerge.id_probability(Trackster::ParticleType::electron) <
           0.5;
  };
  for (size_t i = 0; i < resultTrackstersMerged->size(); i++) {
    auto const &tm = (*resultTrackstersMerged)[i];
    auto &cand = (*resultCandidates)[i];
    //common properties
    cand.setIdProbabilities(tm.id_probabilities());
    //charged candidates
    if (!cand.trackPtr().isNull()) {
      auto pdgId = isHad(tm) ? 211 : 11;
      auto const &tk = cand.trackPtr().get();
      cand.setPdgId(pdgId * tk->charge());
      cand.setCharge(tk->charge());
      cand.setRawEnergy(tm.raw_energy());
      auto const &regrE = tm.regressed_energy();
      math::XYZTLorentzVector p4(regrE * tk->momentum().unit().x(),
                                 regrE * tk->momentum().unit().y(),
                                 regrE * tk->momentum().unit().z(),
                                 regrE);
      cand.setP4(p4);
    } else {  // neutral candidates
      auto pdgId = isHad(tm) ? 130 : 22;
      cand.setPdgId(pdgId);
      cand.setCharge(0);
      cand.setRawEnergy(tm.raw_energy());
      const float &regrE = tm.regressed_energy();
      math::XYZTLorentzVector p4(regrE * tm.barycenter().unit().x(),
                                 regrE * tm.barycenter().unit().y(),
                                 regrE * tm.barycenter().unit().z(),
                                 regrE);
      cand.setP4(p4);
    }
  }
  for (auto &cand : *resultFromTracks) {  //Tracks with no linked tracksters are promoted to charged hadron candidates
    auto const &tk = cand.trackPtr().get();
    cand.setPdgId(211 * tk->charge());
    cand.setCharge(tk->charge());
    const float energy = std::sqrt(tk->p() * tk->p() + ticl::mpion2);
    cand.setRawEnergy(energy);
    math::PtEtaPhiMLorentzVector p4Polar(tk->pt(), tk->eta(), tk->phi(), ticl::mpion);
    cand.setP4(p4Polar);
  }
  // Compute timing
  resultCandidates->insert(resultCandidates->end(), resultFromTracks->begin(), resultFromTracks->end());
  assignTimeToCandidates(*resultCandidates);
  if (regressionAndPid_) {
    // Run inference algorithm
    inferenceAlgo_->inputData(layerClusters, *resultTrackstersMerged, rhtools_);
    inferenceAlgo_->runInference(
        *resultTrackstersMerged);  //option to use "Linking" instead of "CLU3D"/"energyAndPid" instead of "PID"
  }

  evt.put(std::move(resultTrackstersMerged));
  evt.put(std::move(resultCandidates));
}

void TrackstersMergeProducer::assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates) const {
  for (auto &cand : resultCandidates) {
    if (cand.tracksters().size() > 1) {  // For single-trackster candidates the timing is already set
      float time = 0.f;
      float invTimeErr = 0.f;
      for (const auto &tr : cand.tracksters()) {
        if (tr->timeError() > 0) {
          auto invTimeESq = pow(tr->timeError(), -2);
          time += tr->time() * invTimeESq;
          invTimeErr += invTimeESq;
        }
      }
      if (invTimeErr > 0) {
        cand.setTime(time / invTimeErr, sqrt(1.f / invTimeErr));
      }
    }
  }
}

void TrackstersMergeProducer::printTrackstersDebug(const std::vector<Trackster> &tracksters, const char *label) const {
#ifdef EDM_ML_DEBUG
  int counter = 0;
  for (auto const &t : tracksters) {
    LogDebug("TrackstersMergeProducer")
        << counter++ << " TrackstersMergeProducer (" << label << ") obj barycenter: " << t.barycenter()
        << " eta,phi (baricenter): " << t.barycenter().eta() << ", " << t.barycenter().phi()
        << " eta,phi (eigen): " << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi()
        << " pt(eigen): " << std::sqrt(t.eigenvectors(0).Unit().perp2()) * t.raw_energy() << " seedID: " << t.seedID()
        << " seedIndex: " << t.seedIndex() << " size: " << t.vertices().size() << " average usage: "
        << (std::accumulate(std::begin(t.vertex_multiplicity()), std::end(t.vertex_multiplicity()), 0.) /
            (float)t.vertex_multiplicity().size())
        << " raw_energy: " << t.raw_energy() << " regressed energy: " << t.regressed_energy()
        << " probs(ga/e/mu/np/cp/nh/am/unk): ";
    for (auto const &p : t.id_probabilities()) {
      LogDebug("TrackstersMergeProducer") << std::fixed << p << " ";
    }
    LogDebug("TrackstersMergeProducer") << " sigmas: ";
    for (auto const &s : t.sigmas()) {
      LogDebug("TrackstersMergeProducer") << s << " ";
    }
    LogDebug("TrackstersMergeProducer") << std::endl;
  }
#endif
}

void TrackstersMergeProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription linkingDesc;
  linkingDesc.addNode(edm::PluginDescription<LinkingAlgoFactory>("type", "LinkingAlgoByDirectionGeometric", true));
  desc.add<edm::ParameterSetDescription>("linkingPSet", linkingDesc);
  edm::ParameterSetDescription inferenceDesc;
  inferenceDesc.addNode(
      edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByCNNv4", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByCNNv4", inferenceDesc);

  desc.add<edm::InputTag>("trackstersclue3d", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("tracksTime", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("tracksTimeQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tracksTimeErr", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<bool>("regressionAndPid", true);
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<bool>("optimiseAcrossTracksters", true);
  desc.add<bool>("useMTDTiming", true);
  desc.add<int>("eta_bin_window", 1);
  desc.add<int>("phi_bin_window", 1);
  desc.add<double>("pt_sigma_high", 2.);
  desc.add<double>("pt_sigma_low", 2.);
  desc.add<double>("halo_max_distance2", 4.);
  desc.add<double>("track_min_pt", 1.);
  desc.add<double>("track_min_eta", 1.48);
  desc.add<double>("track_max_eta", 3.);
  desc.add<int>("track_max_missing_outerhits", 5);
  desc.add<double>("cosangle_align", 0.9945);
  desc.add<double>("e_over_h_threshold", 1.);
  desc.add<double>("pt_neutral_threshold", 2.);
  desc.add<double>("resol_calo_offset_had", 1.5);
  desc.add<double>("resol_calo_scale_had", 0.15);
  desc.add<double>("resol_calo_offset_em", 1.5);
  desc.add<double>("resol_calo_scale_em", 0.15);
  desc.add<std::string>("inferenceAlgo", "TracksterInferenceByCNNv4");
  descriptions.add("trackstersMergeProducer", desc);
}

DEFINE_FWK_MODULE(TrackstersMergeProducer);
