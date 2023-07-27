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

#include "RecoHGCal/TICL/interface/GlobalCache.h"

#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"

#include "RecoHGCal/TICL/plugins/LinkingAlgoBase.h"
#include "RecoHGCal/TICL/plugins/LinkingAlgoFactory.h"
#include "RecoHGCal/TICL/plugins/LinkingAlgoByDirectionGeometric.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

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

class TrackstersMergeProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersMergeProducer(const edm::ParameterSet &ps);
  ~TrackstersMergeProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<TrackstersCache> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(TrackstersCache *);

  void beginJob();
  void endJob();

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;

private:
  typedef ticl::Trackster::IterationIndex TracksterIterIndex;
  typedef math::XYZVector Vector;

  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, TracksterIterIndex);

  void energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                             const tensorflow::Session *,
                             std::vector<Trackster> &result) const;
  void printTrackstersDebug(const std::vector<Trackster> &, const char *label) const;
  void assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates) const;
  void dumpTrackster(const Trackster &) const;

  std::unique_ptr<LinkingAlgoBase> linkingAlgo_;

  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_clue3d_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_quality_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_err_token_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const std::string tfDnnLabel_;
  const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
  const tensorflow::Session *tfSession_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const std::string detector_;
  const std::string propName_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const bool optimiseAcrossTracksters_;
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
  const std::string eidInputName_;
  const std::string eidOutputNameEnergy_;
  const std::string eidOutputNameId_;
  const float eidMinClusterEnergy_;
  const int eidNLayers_;
  const int eidNClusters_;
  std::once_flag initializeGeometry_;

  const HGCalDDDConstants *hgcons_;

  std::unique_ptr<GeomDet> firstDisk_[2];

  tensorflow::Session *eidSession_;
  hgcal::RecHitTools rhtools_;

  static constexpr int eidNFeatures_ = 3;

  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
};

TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps)
    : tracksters_clue3d_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersclue3d"))),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      tracks_time_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTime"))),
      tracks_time_quality_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeQual"))),
      tracks_time_err_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeErr"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      tfDnnLabel_(ps.getParameter<std::string>("tfDnnLabel")),
      tfDnnToken_(esConsumes(edm::ESInputTag("", tfDnnLabel_))),
      tfSession_(nullptr),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      optimiseAcrossTracksters_(ps.getParameter<bool>("optimiseAcrossTracksters")),
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
      resol_calo_scale_em_(ps.getParameter<double>("resol_calo_scale_em")),
      eidInputName_(ps.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(ps.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(ps.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(ps.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(ps.getParameter<int>("eid_n_layers")),
      eidNClusters_(ps.getParameter<int>("eid_n_clusters")),
      eidSession_(nullptr) {
  produces<std::vector<Trackster>>();
  produces<std::vector<TICLCandidate>>();

  std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  hdc_token_ =
      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));

  auto linkingPSet = ps.getParameter<edm::ParameterSet>("linkingPSet");
  auto algoType = linkingPSet.getParameter<std::string>("type");
  linkingAlgo_ = LinkingAlgoFactory::get()->create(algoType, linkingPSet);
}

void TrackstersMergeProducer::beginJob() {}

void TrackstersMergeProducer::endJob(){};

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
  tfSession_ = es.getData(tfDnnToken_).getSession();

  edm::Handle<std::vector<Trackster>> trackstersclue3d_h;
  evt.getByToken(tracksters_clue3d_token_, trackstersclue3d_h);

  edm::Handle<std::vector<reco::Track>> track_h;
  evt.getByToken(tracks_token_, track_h);
  const auto &tracks = *track_h;

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);
  const auto &muons = evt.get(muons_token_);
  const auto &trackTime = evt.get(tracks_time_token_);
  const auto &trackTimeErr = evt.get(tracks_time_err_token_);
  const auto &trackTimeQual = evt.get(tracks_time_quality_token_);

  // Linking
  linkingAlgo_->linkTracksters(
      track_h, trackTime, trackTimeErr, trackTimeQual, muons, trackstersclue3d_h, *resultCandidates, *resultFromTracks);

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
    outTrackster.setTrackIdx(track_idx);
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
                        rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z());
  energyRegressionAndID(layerClusters, tfSession_, *resultTrackstersMerged);

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

  evt.put(std::move(resultTrackstersMerged));
  evt.put(std::move(resultCandidates));
}

void TrackstersMergeProducer::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                    const tensorflow::Session *eidSession,
                                                    std::vector<Trackster> &tracksters) const {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) threshold. Inference is not applied for soft tracksters.
  // 3. When no trackster passes the selection, return.
  // 4. Create input and output tensors. The batch dimension is determined by the number of
  //    selected tracksters.
  // 5. Fill input tensors with layer cluster features. Per layer, clusters are ordered descending
  //    by energy. Given that tensor data is contiguous in memory, we can use pointer arithmetic to
  //    fill values, even with batching.
  // 6. Zero-fill features for empty clusters in each layer.
  // 7. Batched inference.
  // 8. Assign the regressed energy and id probabilities to each trackster.
  //
  // Indices used throughout this method:
  // i -> batch element / trackster
  // j -> layer
  // k -> cluster
  // l -> feature

  // do nothing when no trackster passes the selection (3)
  int batchSize = (int)tracksters.size();
  if (batchSize == 0) {
    return;
  }

  for (auto &t : tracksters) {
    t.setRegressedEnergy(0.f);
    t.zeroProbabilities();
  }

  // create input and output tensors (4)
  tensorflow::TensorShape shape({batchSize, eidNLayers_, eidNClusters_, eidNFeatures_});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::NamedTensorList inputList = {{eidInputName_, input}};
  static constexpr int inputDimension = 4;

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> outputNames;
  if (!eidOutputNameEnergy_.empty()) {
    outputNames.push_back(eidOutputNameEnergy_);
  }
  if (!eidOutputNameId_.empty()) {
    outputNames.push_back(eidOutputNameId_);
  }

  // fill input tensor (5)
  for (int i = 0; i < batchSize; i++) {
    const Trackster &trackster = tracksters[i];

    // per layer, we only consider the first eidNClusters_ clusters in terms of
    // energy, so in order to avoid creating large / nested structures to do
    // the sorting for an unknown number of total clusters, create a sorted
    // list of layer cluster indices to keep track of the filled clusters
    std::vector<int> clusterIndices(trackster.vertices().size());
    for (int k = 0; k < (int)trackster.vertices().size(); k++) {
      clusterIndices[k] = k;
    }
    sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int &a, const int &b) {
      return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
    });

    // keep track of the number of seen clusters per layer
    std::vector<int> seenClusters(eidNLayers_);

    // loop through clusters by descending energy
    for (const int &k : clusterIndices) {
      // get features per layer and cluster and store the values directly in the input tensor
      const reco::CaloCluster &cluster = layerClusters[trackster.vertices(k)];
      int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
      if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
        // get the pointer to the first feature value for the current batch, layer and cluster
        float *features = &input.tensor<float, inputDimension>()(i, j, seenClusters[j], 0);

        // fill features
        *(features++) = float(cluster.energy() / float(trackster.vertex_multiplicity(k)));
        *(features++) = float(std::abs(cluster.eta()));
        *(features) = float(cluster.phi());

        // increment seen clusters
        seenClusters[j]++;
      }
    }

    // zero-fill features of empty clusters in each layer (6)
    for (int j = 0; j < eidNLayers_; j++) {
      for (int k = seenClusters[j]; k < eidNClusters_; k++) {
        float *features = &input.tensor<float, inputDimension>()(i, j, k, 0);
        for (int l = 0; l < eidNFeatures_; l++) {
          *(features++) = 0.f;
        }
      }
    }
  }

  // run the inference (7)
  tensorflow::run(eidSession, inputList, outputNames, &outputs);

  // store regressed energy per trackster (8)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (int i = 0; i < batchSize; ++i) {
      float regressedEnergy =
          tracksters[i].raw_energy() > eidMinClusterEnergy_ ? energy[i] : tracksters[i].raw_energy();
      tracksters[i].setRegressedEnergy(regressedEnergy);
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = !eidOutputNameEnergy_.empty();
    float *probs = outputs[probsIdx].flat<float>().data();
    int probsNumber = tracksters[0].id_probabilities().size();
    for (int i = 0; i < batchSize; ++i) {
      tracksters[i].setProbabilities(&probs[i * probsNumber]);
    }
  }
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
        cand.setTime(time / invTimeErr);
        cand.setTimeError(sqrt(1.f / invTimeErr));
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

  desc.add<edm::InputTag>("trackstersclue3d", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("tracksTime", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("tracksTimeQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tracksTimeErr", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<bool>("optimiseAcrossTracksters", true);
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
  desc.add<std::string>("tfDnnLabel", "tracksterSelectionTf");
  desc.add<std::string>("eid_input_name", "input");
  desc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  desc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  desc.add<double>("eid_min_cluster_energy", 2.5);
  desc.add<int>("eid_n_layers", 50);
  desc.add<int>("eid_n_clusters", 10);
  descriptions.add("trackstersMergeProducer", desc);
}

DEFINE_FWK_MODULE(TrackstersMergeProducer);
