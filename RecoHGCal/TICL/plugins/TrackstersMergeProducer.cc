#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoHGCal/TICL/plugins/GlobalCache.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "TrackstersPCA.h"

using namespace ticl;

class TrackstersMergeProducer : public edm::stream::EDProducer<edm::GlobalCache<TrackstersCache>> {
public:
  explicit TrackstersMergeProducer(const edm::ParameterSet &ps, const CacheBase *cache);
  ~TrackstersMergeProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters, std::vector<Trackster> &result) const;

  // static methods for handling the global cache
  static std::unique_ptr<TrackstersCache> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(TrackstersCache *);

private:
  enum TracksterIterIndex { EM = 0, TRK, HAD, SEED };

  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, TracksterIterIndex);

  void printTrackstersDebug(const std::vector<Trackster> &, const char *label) const;
  void dumpTrackster(const Trackster &) const;

  const edm::EDGetTokenT<std::vector<Trackster>> trackstersem_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> tracksterstrk_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> trackstershad_token_;
  const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seedingTrk_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const bool optimiseAcrossTracksters_;
  const int eta_bin_window_;
  const int phi_bin_window_;
  const double pt_sigma_high_;
  const double pt_sigma_low_;
  const double cosangle_align_;
  const double e_over_h_threshold_;
  const double pt_neutral_threshold_;
  const double resol_calo_offset_;
  const double resol_calo_scale_;
  const bool debug_;
  const std::string eidInputName_;
  const std::string eidOutputNameEnergy_;
  const std::string eidOutputNameId_;
  const float eidMinClusterEnergy_;
  const int eidNLayers_;
  const int eidNClusters_;

  tensorflow::Session *eidSession_;
  hgcal::RecHitTools rhtools_;

  static constexpr int eidNFeatures_ = 3;
};

TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps, const CacheBase *cache)
    : trackstersem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersem"))),
      tracksterstrk_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksterstrk"))),
      trackstershad_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstershad"))),
      seedingTrk_token_(consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seedingTrk"))),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      optimiseAcrossTracksters_(ps.getParameter<bool>("optimiseAcrossTracksters")),
      eta_bin_window_(ps.getParameter<int>("eta_bin_window")),
      phi_bin_window_(ps.getParameter<int>("phi_bin_window")),
      pt_sigma_high_(ps.getParameter<double>("pt_sigma_high")),
      pt_sigma_low_(ps.getParameter<double>("pt_sigma_low")),
      cosangle_align_(ps.getParameter<double>("cosangle_align")),
      e_over_h_threshold_(ps.getParameter<double>("e_over_h_threshold")),
      pt_neutral_threshold_(ps.getParameter<double>("pt_neutral_threshold")),
      resol_calo_offset_(ps.getParameter<double>("resol_calo_offset")),
      resol_calo_scale_(ps.getParameter<double>("resol_calo_scale")),
      debug_(ps.getParameter<bool>("debug")),
      eidInputName_(ps.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(ps.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(ps.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(ps.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(ps.getParameter<int>("eid_n_layers")),
      eidNClusters_(ps.getParameter<int>("eid_n_clusters")),
      eidSession_(nullptr) {
  // mount the tensorflow graph onto the session when set
  const TrackstersCache *trackstersCache = dynamic_cast<const TrackstersCache *>(cache);
  if (trackstersCache == nullptr || trackstersCache->eidGraphDef == nullptr) {
    throw cms::Exception("MissingGraphDef")
        << "TrackstersMergeProducer received an empty graph definition from the global cache";
  }
  eidSession_ = tensorflow::createSession(trackstersCache->eidGraphDef);

  produces<std::vector<Trackster>>();
}

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
  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);
  auto result = std::make_unique<std::vector<Trackster>>();
  auto mergedTrackstersTRK = std::make_unique<std::vector<Trackster>>();

  TICLTracksterTiles tracksterTile;
  std::vector<bool> usedTrackstersEM;
  std::vector<bool> usedTrackstersTRK;
  std::vector<bool> usedTrackstersHAD;
  std::vector<bool> usedSeeds;

  edm::Handle<std::vector<reco::Track>> track_h;
  evt.getByToken(tracks_token_, track_h);
  const auto &tracks = *track_h;

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  evt.getByToken(clusters_token_, cluster_h);
  const auto &layerClusters = *cluster_h;

  edm::Handle<std::vector<Trackster>> trackstersem_h;
  evt.getByToken(trackstersem_token_, trackstersem_h);
  const auto &trackstersEM = *trackstersem_h;
  usedTrackstersEM.resize(trackstersEM.size(), false);

  edm::Handle<std::vector<Trackster>> tracksterstrk_h;
  evt.getByToken(tracksterstrk_token_, tracksterstrk_h);
  const auto &trackstersTRK = *tracksterstrk_h;
  usedTrackstersTRK.resize(trackstersTRK.size(), false);

  edm::Handle<std::vector<Trackster>> trackstershad_h;
  evt.getByToken(trackstershad_token_, trackstershad_h);
  const auto &trackstersHAD = *trackstershad_h;
  usedTrackstersHAD.resize(trackstersHAD.size(), false);

  edm::Handle<std::vector<TICLSeedingRegion>> seedingTrk_h;
  evt.getByToken(seedingTrk_token_, seedingTrk_h);
  const auto &seedingTrk = *seedingTrk_h;
  usedSeeds.resize(seedingTrk.size(), false);

  fillTile(tracksterTile, trackstersEM, TracksterIterIndex::EM);
  fillTile(tracksterTile, trackstersTRK, TracksterIterIndex::TRK);
  fillTile(tracksterTile, trackstersHAD, TracksterIterIndex::HAD);

  auto seedId = 0;
  for (auto const &s : seedingTrk) {
    tracksterTile.fill(TracksterIterIndex::SEED, s.origin.eta(), s.origin.phi(), seedId++);

    if (debug_) {
      LogDebug("TrackstersMergeProducer")
          << "Seed index: " << seedId << " internal index: " << s.index << " origin: " << s.origin
          << " mom: " << s.directionAtOrigin << " pt: " << std::sqrt(s.directionAtOrigin.perp2())
          << " zSide: " << s.zSide << " collectionID: " << s.collectionID << " track pt " << tracks[s.index].pt()
          << std::endl;
    }
  }

  if (debug_) {
    printTrackstersDebug(trackstersTRK, "tracksterTRK");
    printTrackstersDebug(trackstersEM, "tracksterEM");
    printTrackstersDebug(trackstersHAD, "tracksterHAD");
  }

  int tracksterTRK_idx = 0;
  int tracksterHAD_idx = 0;
  if (optimiseAcrossTracksters_) {
    for (auto const &t : trackstersTRK) {
      if (debug_) {
        int entryEtaBin = tracksterTile[TracksterIterIndex::TRK].etaBin(t.barycenter().eta());
        int entryPhiBin = tracksterTile[TracksterIterIndex::TRK].phiBin(t.barycenter().phi());
        int bin = tracksterTile[TracksterIterIndex::TRK].globalBin(t.barycenter().eta(), t.barycenter().phi());
        LogDebug("TrackstersMergeProducer")
            << "TrackstersMergeProducer Tracking obj: " << t.barycenter() << " in bin " << bin << " etaBin "
            << entryEtaBin << " phiBin " << entryPhiBin << std::endl;
        dumpTrackster(t);
      }
      auto const &track = tracks[t.seedIndex()];
      auto trk_pt = (float)track.pt();
      auto diff_pt = t.raw_pt() - trk_pt;
      auto pt_err = trk_pt * resol_calo_scale_ + resol_calo_offset_;
      auto w_cal = 1. / (pt_err * pt_err);
      auto w_trk = 1. / (track.ptError() * track.ptError());
      auto diff_pt_sigmas = diff_pt / pt_err;
      auto e_over_h = (t.raw_em_pt() / ((t.raw_pt() - t.raw_em_pt()) != 0. ? (t.raw_pt() - t.raw_em_pt()) : 1.));
      LogDebug("TrackstersMergeProducer")
          << "trackster_pt: " << t.raw_pt() << std::endl
          << "track pt   (inner): " << track.pt() << std::endl
          << "track eta  (inner): " << track.eta() << std::endl
          << "track _phi (inner): " << track.phi() << std::endl
          << "track pt   (outer): " << std::sqrt(track.outerMomentum().perp2()) << std::endl
          << "track eta  (outer): " << track.outerMomentum().eta() << std::endl
          << "track _phi (outer): " << track.outerMomentum().phi() << std::endl
          << "pt_err_track: " << track.ptError() << std::endl
          << "diff_pt: " << diff_pt << std::endl
          << "pt_err: " << pt_err << std::endl
          << "diff_pt_sigmas: " << diff_pt_sigmas << std::endl
          << "w_cal: " << w_cal << std::endl
          << "w_trk: " << w_trk << std::endl
          << "average_pt: " << (t.raw_pt() * w_cal + trk_pt * w_trk) / (w_cal + w_trk) << std::endl
          << "e_over_h: " << e_over_h << std::endl;

      // If the energy is unbalanced and higher in Calo ==> balance it by
      // emitting gammas/neutrals
      if (diff_pt_sigmas > pt_sigma_high_) {
        if (e_over_h > e_over_h_threshold_) {
          auto gamma_pt = std::min(diff_pt, t.raw_em_pt());
          // Create gamma with calo direction
          LogDebug("TrackstersMergeProducer")
              << " Creating a photon from TRK Trackster with energy " << gamma_pt << " and direction "
              << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
          diff_pt -= gamma_pt;
          if (diff_pt > pt_neutral_threshold_) {
            // Create also a neutral on top, with calo direction and diff_pt as energy
            LogDebug("TrackstersMergeProducer")
                << " Adding also a neutral hadron from TRK Trackster with energy " << diff_pt << " and direction "
                << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
          }
        } else {
          // Create neutral with calo direction
          LogDebug("TrackstersMergeProducer")
              << " Creating a neutral hadron from TRK Trackster with energy " << diff_pt << " and direction "
              << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
        }
      }
      // If the energy is in the correct ball-park (this also covers the
      // previous case after the neutral emission), use weighted averages
      if (diff_pt_sigmas > -pt_sigma_low_) {
        // Create either an electron or a charged hadron, using the weighted
        // average information between the track and the cluster for the
        // energy, the direction of the track at the vertex.  The PID is simply
        // passed over to the final trackster, while the energy is changed.
        auto average_pt = (w_cal * t.raw_pt() + trk_pt * w_trk) / (w_cal + w_trk);
        LogDebug("TrackstersMergeProducer")
            << " Creating electron/charged hadron from TRK Trackster with weighted p_t " << average_pt
            << " and direction " << track.eta() << ", " << track.phi() << std::endl;

      }
      // If the energy of the calo is too low, just use track-only information
      else {
        // Create either an electron or a charged hadron, using the track
        // information only.
        LogDebug("TrackstersMergeProducer")
            << " Creating electron/charged hadron from TRK Trackster with track p_t " << trk_pt << " and direction "
            << track.eta() << ", " << track.phi() << std::endl;
      }
      result->push_back(t);
      usedTrackstersTRK[tracksterTRK_idx] = true;
      tracksterTRK_idx++;
    }
  }

  tracksterTRK_idx = 0;
  for (auto const &t : trackstersTRK) {
    if (debug_) {
      LogDebug("TrackstersMergeProducer") << " Considering trackster " << tracksterTRK_idx
                                          << " as used: " << usedTrackstersTRK[tracksterTRK_idx] << std::endl;
    }
    if (!usedTrackstersTRK[tracksterTRK_idx]) {
      LogDebug("TrackstersMergeProducer")
          << " Creating a charge hadron from TRK Trackster with track energy " << t.raw_energy() << " and direction "
          << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
      result->push_back(t);
    }
    tracksterTRK_idx++;
  }

  auto tracksterEM_idx = 0;
  for (auto const &t : trackstersEM) {
    if (debug_) {
      LogDebug("TrackstersMergeProducer") << " Considering trackster " << tracksterEM_idx
                                          << " as used: " << usedTrackstersEM[tracksterEM_idx] << std::endl;
    }
    if (!usedTrackstersEM[tracksterEM_idx]) {
      LogDebug("TrackstersMergeProducer")
          << " Creating a photon from EM Trackster with track energy " << t.raw_energy() << " and direction "
          << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
      result->push_back(t);
    }
    tracksterEM_idx++;
  }

  tracksterHAD_idx = 0;
  for (auto const &t : trackstersHAD) {
    if (debug_) {
      LogDebug("TrackstersMergeProducer") << " Considering trackster " << tracksterHAD_idx
                                          << " as used: " << usedTrackstersHAD[tracksterHAD_idx] << std::endl;
    }
    if (!usedTrackstersHAD[tracksterHAD_idx]) {
      LogDebug("TrackstersMergeProducer")
          << " Creating a neutral hadron from HAD Trackster with track energy " << t.raw_energy() << " and direction "
          << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi() << std::endl;
      result->push_back(t);
    }
    tracksterHAD_idx++;
  }

  assignPCAtoTracksters(*result, layerClusters, rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z());
  energyRegressionAndID(layerClusters, *result);

  printTrackstersDebug(*result, "TrackstersMergeProducer");

  evt.put(std::move(result));
}

void TrackstersMergeProducer::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                    std::vector<Trackster> &tracksters) const {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) treshold. Inference is not applied for soft tracksters.
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

  // set default values per trackster, determine if the cluster energy threshold is passed,
  // and store indices of hard tracksters
  std::vector<int> tracksterIndices;
  for (int i = 0; i < (int)tracksters.size(); i++) {
    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the threshold
    // which is enough to decide whether to run inference for the trackster or
    // not
    float sumClusterEnergy = 0.;
    for (const unsigned int &vertex : tracksters[i].vertices()) {
      sumClusterEnergy += (float)layerClusters[vertex].energy();
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy >= eidMinClusterEnergy_) {
        // set default values (1)
        tracksters[i].setRegressedEnergy(0.f);
        tracksters[i].zeroProbabilities();
        tracksterIndices.push_back(i);
        break;
      }
    }
  }

  // do nothing when no trackster passes the selection (3)
  int batchSize = (int)tracksterIndices.size();
  if (batchSize == 0) {
    return;
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
    const Trackster &trackster = tracksters[tracksterIndices[i]];

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
  tensorflow::run(eidSession_, inputList, outputNames, &outputs);

  // store regressed energy per trackster (8)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setRegressedEnergy(*(energy++));
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = !eidOutputNameEnergy_.empty();
    float *probs = outputs[probsIdx].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setProbabilities(probs);
      probs += tracksters[i].id_probabilities().size();
    }
  }
}

std::unique_ptr<TrackstersCache> TrackstersMergeProducer::initializeGlobalCache(const edm::ParameterSet &params) {
  // this method is supposed to create, initialize and return a TrackstersCache instance
  std::unique_ptr<TrackstersCache> cache = std::make_unique<TrackstersCache>(params);

  // load the graph def and save it
  std::string graphPath = params.getParameter<std::string>("eid_graph_path");
  if (!graphPath.empty()) {
    graphPath = edm::FileInPath(graphPath).fullPath();
    cache->eidGraphDef = tensorflow::loadGraphDef(graphPath);
  }

  return cache;
}

void TrackstersMergeProducer::globalEndJob(TrackstersCache *cache) {
  delete cache->eidGraphDef;
  cache->eidGraphDef = nullptr;
}

void TrackstersMergeProducer::printTrackstersDebug(const std::vector<Trackster> &tracksters, const char *label) const {
  if (!debug_)
    return;

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
}

void TrackstersMergeProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersem", edm::InputTag("ticlTrackstersEM"));
  desc.add<edm::InputTag>("tracksterstrk", edm::InputTag("ticlTrackstersTrk"));
  desc.add<edm::InputTag>("trackstershad", edm::InputTag("ticlTrackstersHAD"));
  desc.add<edm::InputTag>("seedingTrk", edm::InputTag("ticlSeedingTrk"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<bool>("optimiseAcrossTracksters", true);
  desc.add<int>("eta_bin_window", 1);
  desc.add<int>("phi_bin_window", 1);
  desc.add<double>("pt_sigma_high", 2.);
  desc.add<double>("pt_sigma_low", 2.);
  desc.add<double>("cosangle_align", 0.9945);
  desc.add<double>("e_over_h_threshold", 1.);
  desc.add<double>("pt_neutral_threshold", 2.);
  desc.add<double>("resol_calo_offset", 1.5);
  desc.add<double>("resol_calo_scale", 0.15);
  desc.add<bool>("debug", true);
  desc.add<std::string>("eid_graph_path", "RecoHGCal/TICL/data/tf_models/energy_id_v0.pb");
  desc.add<std::string>("eid_input_name", "input");
  desc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  desc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  desc.add<double>("eid_min_cluster_energy", 1.);
  desc.add<int>("eid_n_layers", 50);
  desc.add<int>("eid_n_clusters", 10);
  descriptions.add("trackstersMergeProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackstersMergeProducer);
