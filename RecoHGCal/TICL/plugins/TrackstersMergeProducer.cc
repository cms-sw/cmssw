// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 11/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoHGCal/TICL/plugins/GlobalCache.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "TrackstersPCA.h"

using namespace ticl;

class TrackstersMergeProducer : public edm::stream::EDProducer<edm::GlobalCache<TrackstersCache>> {
public:
  explicit TrackstersMergeProducer(const edm::ParameterSet &ps, const CacheBase* cache);
  ~TrackstersMergeProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void energyRegressionAndID(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& result) const;

  // static methods for handling the global cache
  static std::unique_ptr<TrackstersCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(TrackstersCache*);

private:
  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, int);

  void printTrackstersDebug(const std::vector<Trackster> &, const char * label) const;
  void mergeTrackstersTRK(const std::vector<Trackster> &,
      const std::vector<reco::CaloCluster>&, std::vector<Trackster> &) const;

  const edm::EDGetTokenT<std::vector<Trackster>> trackstersem_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> tracksterstrk_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> trackstershad_token_;
  const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seedingTrk_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const bool oneTracksterPerTrackSeed_;
  const std::string eidInputName_;
  const std::string eidOutputNameEnergy_;
  const std::string eidOutputNameId_;
  const float eidMinClusterEnergy_;
  const int eidNLayers_;
  const int eidNClusters_;

  tensorflow::Session* eidSession_;
  hgcal::RecHitTools rhtools_;

  static const int eidNFeatures_ = 3;
  static constexpr float trkptOffset_ = 1.5;
  static constexpr float trkptScale_ = 0.15;
};


TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps, const CacheBase *cache) :
  trackstersem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersem"))),
  tracksterstrk_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksterstrk"))),
  trackstershad_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstershad"))),
  seedingTrk_token_(consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seedingTrk"))),
  clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
  oneTracksterPerTrackSeed_(ps.getParameter<bool>("oneTracksterPerTrackSeed")),
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

void TrackstersMergeProducer::fillTile(TICLTracksterTiles & tracksterTile,
    const std::vector<Trackster> & tracksters,
    int tracksterIteration) {
  int tracksterId = 0;
  for (auto const & t: tracksters) {
    tracksterTile.fill(tracksterIteration, t.barycenter.eta(), t.barycenter.phi(), tracksterId);
    LogDebug("TrackstersMergeProducer") << "Adding tracksterId: " << tracksterId << " into bin [eta,phi]: [ "
                                      << tracksterTile[tracksterIteration].etaBin(t.barycenter.eta())
                                      << ", " << tracksterTile[tracksterIteration].phiBin(t.barycenter.phi())
                                      << "] for iteration: " << tracksterIteration << std::endl;

    tracksterId++;
  }
}

void TrackstersMergeProducer::mergeTrackstersTRK(const std::vector<Trackster> & input,
    const std::vector<reco::CaloCluster> &layerClusters,
    std::vector<Trackster> & output) const {

  std::map<int, int> seedIndices;

  output.reserve(input.size());

  for (auto const  & t : input) {
    if (seedIndices.find(t.seedIndex) != seedIndices.end()) {
      std::cout << "Seed index: " << t.seedIndex
        << " already used by trackster: " << seedIndices[t.seedIndex]
        << std::endl;

      auto & old = output[seedIndices[t.seedIndex]];
      auto updated_size = old.vertices.size();
      std::cout << "Old size: " << updated_size << std::endl;
      updated_size += t.vertices.size();
      std::cout << "Updatd size: " << updated_size << std::endl;
      old.vertices.reserve(updated_size);
      old.vertex_multiplicity.reserve(updated_size);
      std::copy(std::begin(t.vertices),
          std::end(t.vertices),
          std::back_inserter(old.vertices)
          );
      std::copy(std::begin(t.vertex_multiplicity),
          std::end(t.vertex_multiplicity),
          std::back_inserter(old.vertex_multiplicity)
          );
    } else {
      std::cout << "Passing down trackster " << output.size()
        << " with seedIndex: " << t.seedIndex << std::endl;
      seedIndices[t.seedIndex] = output.size();
      output.push_back(t);
    }
  }

  assignPCAtoTracksters(output, layerClusters,
      rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z());
  energyRegressionAndID(layerClusters, output);
}

void TrackstersMergeProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  rhtools_.getEventSetup(es);
  auto result = std::make_unique<std::vector<Trackster>>();
  auto mergedTrackstersTRK = std::make_unique<std::vector<Trackster>>();

  TICLTracksterTiles tracksterTile;
  std::vector<bool> usedTrackstersEM;
  std::vector<bool> usedTrackstersTRK;
  std::vector<bool> usedTrackstersHAD;
  std::vector<bool> usedSeeds;

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  evt.getByToken(clusters_token_, cluster_h);
  const auto& layerClusters = *cluster_h;

  edm::Handle<std::vector<Trackster>> trackstersem_h;
  evt.getByToken(trackstersem_token_, trackstersem_h);
  const auto &trackstersEM = *trackstersem_h;
  usedTrackstersEM.resize(trackstersEM.size(), 0);

  edm::Handle<std::vector<Trackster>> tracksterstrk_h;
  evt.getByToken(tracksterstrk_token_, tracksterstrk_h);
  const auto &tmp_trackstersTRK = *tracksterstrk_h;
//  usedTrackstersTRK.resize(trackstersTRK.size(), 0);

  edm::Handle<std::vector<Trackster>> trackstershad_h;
  evt.getByToken(trackstershad_token_, trackstershad_h);
  const auto &trackstersHAD = *trackstershad_h;
  usedTrackstersHAD.resize(trackstersHAD.size(), 0);

  edm::Handle<std::vector<TICLSeedingRegion>> seedingTrk_h;
  evt.getByToken(seedingTrk_token_, seedingTrk_h);
  const auto & seedingTrk = * seedingTrk_h;
  usedSeeds.resize(seedingTrk.size(), 0);

  if (oneTracksterPerTrackSeed_) {
    mergeTrackstersTRK(tmp_trackstersTRK, layerClusters, *mergedTrackstersTRK);
    printTrackstersDebug(*mergedTrackstersTRK, "tracksterTRKMERGED");
  }

  const auto &trackstersTRK = *mergedTrackstersTRK;
  usedTrackstersTRK.resize(trackstersTRK.size(), 0);

  int tracksterIteration = 0;
  fillTile(tracksterTile, trackstersEM, tracksterIteration++);
  fillTile(tracksterTile, trackstersTRK, tracksterIteration++);
  fillTile(tracksterTile, trackstersHAD, tracksterIteration++);

  auto seedId = 0;
  for (auto const & s : seedingTrk) {
    std::cout << "Seed index: " << seedId
      << " internal index: " << s.index
      << " origin: " << s.origin
      << " mom: " << s.directionAtOrigin
      << " zSide: " << s.zSide
      << " collectionID: " << s.collectionID
      << std::endl;
    tracksterTile.fill(tracksterIteration, s.origin.eta(), s.origin.phi(), seedId++);
  }

  if (1) {
    printTrackstersDebug(trackstersTRK, "tracksterTRK");
    printTrackstersDebug(trackstersEM, "tracksterEM");
    printTrackstersDebug(trackstersHAD, "tracksterHAD");
  }

  int tracksterTRK_idx = 0;
  for (auto const & t : trackstersTRK) {
    int entryEtaBin = tracksterTile[3].etaBin(t.barycenter.eta());
    int entryPhiBin = tracksterTile[3].phiBin(t.barycenter.phi());
    int deltaIEta = 1;
    int deltaIPhi = 1;
    int nEtaBins = 34;
    int nPhiBins = 126;
    int bin = tracksterTile[3].globalBin(t.barycenter.eta(), t.barycenter.phi());
      std::cout << "TrackstersMergeProducer Tracking obj: " << t.barycenter
        << " in bin " << bin
        << " etaBin " << entryEtaBin
        << " phiBin " << entryPhiBin
        << " regressed energy: " << t.regressed_energy
        << " raw_energy: " << t.raw_energy
        << " seedIndex: " << t.seedIndex
        << std::endl;
    auto const & original_seed = seedingTrk[t.seedIndex];
    auto trk_pt = std::sqrt(original_seed.directionAtOrigin.perp2());
    auto diff_pt = t.raw_pt - trk_pt;
    auto pt_err = trk_pt*trkptScale_ + trkptOffset_;
    auto diff_pt_sigmas = diff_pt/pt_err;
    auto e_over_h = (t.raw_em_pt/((t.raw_pt-t.raw_em_pt) != 0. ? (t.raw_pt-t.raw_em_pt) : -1.));
    std::cout << "Original seed pos " << original_seed.origin
      << ", mom: " << original_seed.directionAtOrigin.mag()
      << " calo_pt(" << t.raw_pt << "/" << t.raw_em_pt << "/"
      <<  e_over_h << ") - seed_pt(" << trk_pt << "): " << diff_pt
      << " in sigmas: " << diff_pt/pt_err
      << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(original_seed.directionAtOrigin.unit()))
      << std::endl;
    if (diff_pt_sigmas < -2.) {
      auto startEtaBin = std::max(entryEtaBin - deltaIEta, 0);
      auto endEtaBin = std::min(entryEtaBin + deltaIEta + 1, nEtaBins);
      auto startPhiBin = entryPhiBin - deltaIPhi;
      auto endPhiBin = entryPhiBin + deltaIPhi + 1;
      bool recoverEM = (e_over_h < 1);
      for (int ieta = startEtaBin; ieta < endEtaBin; ++ieta) {
        auto offset = ieta * nPhiBins;
        for (int iphi_it = startPhiBin; iphi_it < endPhiBin; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBins + nPhiBins) % nPhiBins);
          auto ibin = offset + iphi;
          auto const & searchable = recoverEM ? tracksterTile[0][ibin] : tracksterTile[2][ibin];
          auto const & searchableTracksters = recoverEM ? trackstersEM : trackstersHAD;
          auto & searchableUsed = recoverEM ? usedTrackstersEM : usedTrackstersHAD;
          std::cout << "Trying to recover energy (EM?):" << recoverEM << std::endl;
          std::cout << "Candidates in bin " << ibin << " are: " << searchable.size() << std::endl;
          for (auto const & s : searchable) {
            auto const & st = searchableTracksters[s];
            auto cos_angle = std::abs(t.eigenvectors[0].Dot(st.eigenvectors[0]));
            if (cos_angle > 0.9945) {
              usedTrackstersTRK[tracksterTRK_idx] = 1;
              searchableUsed[s] = 1;
              auto combined = t;
              std::copy(std::begin(st.vertices), std::end(st.vertices),
                  std::back_inserter(combined.vertices));
              std::copy(std::begin(st.vertex_multiplicity), std::end(st.vertex_multiplicity),
                  std::back_inserter(combined.vertex_multiplicity));
              std::cout << " linked to st obj: " << st.barycenter
                << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(st.eigenvectors[0]))
                << std::endl
                << " regressed energy: " << st.regressed_energy
                << " raw_energy: " << st.raw_energy
                << " cumulative: " << (t.raw_energy+st.raw_energy)
                << " (trk+st)/seed: " << (t.raw_energy+st.raw_energy)/original_seed.directionAtOrigin.mag()
                << std::endl;
              result->push_back(combined);
            } else {
              std::cout << " Missed link to st obj: " << st.barycenter
                << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(st.eigenvectors[0]))
                << std::endl
                << " regressed energy: " << st.regressed_energy
                << " raw_energy: " << st.raw_energy
                << " cumulative: " << (t.raw_energy+st.raw_energy)
                << " (trk+st)/seed: " << (t.raw_energy+st.raw_energy)/original_seed.directionAtOrigin.mag()
                << std::endl;

            }
          }
        }
      }
    } else if (diff_pt_sigmas > 2.) {
      std::cout << "Object should be split" << std::endl;
    } else {
      result->push_back(t);
      usedTrackstersTRK[tracksterTRK_idx] = 1;
    }
    tracksterTRK_idx++;
  }

  auto tracksterHAD_idx = 0;
  for (auto const & t : trackstersHAD) {
    int bin = tracksterTile[2].globalBin(t.barycenter.eta(), t.barycenter.phi());
      std::cout << "TrackstersMergeProducer HAD obj: " << t.barycenter
        << " regressed energy: " << t.regressed_energy
        << " raw_energy: " << t.raw_energy
        << std::endl;
    auto const & ems = tracksterTile[0][bin];
      std::cout << "Trying to associate closeby em energy" << std::endl;
      auto tracksterEM_idx = 0;
      for (auto const & e : ems) {
        auto const & em = trackstersEM[e];
        auto cos_angle = std::abs(t.eigenvectors[0].Dot(em.eigenvectors[0]));
        if (cos_angle > 0.9945) {
          usedTrackstersHAD[tracksterHAD_idx] = 1;
          usedTrackstersEM[tracksterEM_idx] = 1;
          auto combined = t;
          std::copy(std::begin(em.vertices), std::end(em.vertices),
              std::back_inserter(combined.vertices));
          std::copy(std::begin(em.vertex_multiplicity), std::end(em.vertex_multiplicity),
              std::back_inserter(combined.vertex_multiplicity));
          std::cout << " linked to em obj: " << em.barycenter
            << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(em.eigenvectors[0]))
            << std::endl
            << " regressed energy: " << em.regressed_energy
            << " raw_energy: " << em.raw_energy
            << " cumulative: " << (t.raw_energy+em.raw_energy)
            << std::endl;
          result->push_back(combined);
        } else {
          std::cout << " Missed link to em obj: " << em.barycenter
            << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(em.eigenvectors[0]))
            << std::endl
            << " regressed energy: " << em.regressed_energy
            << " raw_energy: " << em.raw_energy
            << " cumulative: " << (t.raw_energy+em.raw_energy)
            << std::endl;
        }
        tracksterEM_idx++;
      }
    tracksterHAD_idx++;
  }

  tracksterTRK_idx = 0;
  for (auto const & t : trackstersTRK) {
    if (! usedTrackstersTRK[tracksterTRK_idx]) {
      result->push_back(t);
    }
    tracksterTRK_idx++;
  }

  auto tracksterEM_idx = 0;
  for (auto const & t : trackstersEM) {
    if (! usedTrackstersEM[tracksterEM_idx]) {
      result->push_back(t);
    }
    tracksterEM_idx++;
  }

  tracksterHAD_idx = 0;
  for (auto const & t : trackstersHAD) {
    if (! usedTrackstersHAD[tracksterHAD_idx]) {
      result->push_back(t);
    }
    tracksterHAD_idx++;
  }

  assignPCAtoTracksters(*result, layerClusters,
      rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z());
  energyRegressionAndID(layerClusters, *result);

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
    // set default values (1)
    tracksters[i].regressed_energy = 0.;
    for (float &p : tracksters[i].id_probabilities) {
      p = 0.;
    }

    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the threshold which is enough to
    // decide whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for (const unsigned int &vertex : tracksters[i].vertices) {
      sumClusterEnergy += (float)layerClusters[vertex].energy();
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy >= eidMinClusterEnergy_) {
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

    // per layer, we only consider the first eidNClusters_ clusters in terms of energy, so in order
    // to avoid creating large / nested structures to do the sorting for an unknown number of total
    // clusters, create a sorted list of layer cluster indices to keep track of the filled clusters
    std::vector<int> clusterIndices(trackster.vertices.size());
    for (int k = 0; k < (int)trackster.vertices.size(); k++) {
      clusterIndices[k] = k;
    }
    sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int &a, const int &b) {
      return layerClusters[trackster.vertices[a]].energy() > layerClusters[trackster.vertices[b]].energy();
    });

    // keep track of the number of seen clusters per layer
    std::vector<int> seenClusters(eidNLayers_);

    // loop through clusters by descending energy
    for (const int &k : clusterIndices) {
      // get features per layer and cluster and store the values directly in the input tensor
      const reco::CaloCluster &cluster = layerClusters[trackster.vertices[k]];
      int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
      if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
        // get the pointer to the first feature value for the current batch, layer and cluster
        float *features = &input.tensor<float, 4>()(i, j, seenClusters[j], 0);

        // fill features
        *(features++) = float(std::abs(cluster.eta()));
        *(features++) = float(cluster.phi());
        *features = float(cluster.energy());

        // increment seen clusters
        seenClusters[j]++;
      }
    }

    // zero-fill features of empty clusters in each layer (6)
    for (int j = 0; j < eidNLayers_; j++) {
      for (int k = seenClusters[j]; k < eidNClusters_; k++) {
        float *features = &input.tensor<float, 4>()(i, j, k, 0);
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
      tracksters[i].regressed_energy = *(energy++);
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = eidOutputNameEnergy_.empty() ? 0 : 1;
    float *probs = outputs[probsIdx].flat<float>().data();

    for (const int &i : tracksterIndices) {
      for (float &p : tracksters[i].id_probabilities) {
        p = *(probs++);
      }
    }
  }
}

std::unique_ptr<TrackstersCache> TrackstersMergeProducer::initializeGlobalCache(const edm::ParameterSet& params) {
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

void TrackstersMergeProducer::globalEndJob(TrackstersCache* cache) {
  delete cache->eidGraphDef;
  cache->eidGraphDef = nullptr;
}

void TrackstersMergeProducer::printTrackstersDebug(const std::vector<Trackster> & tracksters,
    const char * label) const {
    int counter = 0;
    for (auto const & t : tracksters) {
      std::cout << counter++ << " TrackstersMergeProducer " << label << " obj barycenter: "
        << t.barycenter
        << " eta,phi: " << t.barycenter.eta() << ", " << t.barycenter.phi()
        << " pt: " << std::sqrt(t.eigenvectors[0].Unit().perp2())*t.raw_energy
        << " seedID: " << t.seedID
        << " seedIndex: " << t.seedIndex
        << " size: " << t.vertices.size()
        << " average usage: " <<
        (std::accumulate(
                         std::begin(t.vertex_multiplicity),
                         std::end(t.vertex_multiplicity), 0.)/(float)t.vertex_multiplicity.size())
        << " raw_energy: " << t.raw_energy
        << " regressed energy: " << t.regressed_energy
        << " probs: ";
        for (auto const & p : t.id_probabilities) {
          std::cout << p << " ";
        }
        std::cout << " sigmas: ";
        for (auto const & s : t.sigmas) {
          std::cout << s << " ";
        }
        std::cout << std::endl;
    }
}
void TrackstersMergeProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersem", edm::InputTag("trackstersEM"));
  desc.add<edm::InputTag>("tracksterstrk", edm::InputTag("trackstersTrk"));
  desc.add<edm::InputTag>("trackstershad", edm::InputTag("trackstersHAD"));
  desc.add<edm::InputTag>("seedingTrk", edm::InputTag("ticlSeedingTrk"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<bool>("oneTracksterPerTrackSeed", true);
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
