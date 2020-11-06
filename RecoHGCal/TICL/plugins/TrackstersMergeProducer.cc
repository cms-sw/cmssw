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
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

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
  enum TracksterIterIndex { TRKEM = 0, EM, TRK, HAD };

  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, TracksterIterIndex);

  void printTrackstersDebug(const std::vector<Trackster> &, const char *label) const;
  void dumpTrackster(const Trackster &) const;

  const edm::EDGetTokenT<std::vector<Trackster>> tracksterstrkem_token_;
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
    : tracksterstrkem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksterstrkem"))),
      trackstersem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersem"))),
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
  produces<std::vector<TICLCandidate>>();
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
  auto resultTrackstersMerged = std::make_unique<std::vector<Trackster>>();
  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();

  TICLTracksterTiles tracksterTile;
  std::vector<bool> usedTrackstersMerged;
  std::vector<int> indexInMergedCollTRKEM;
  std::vector<int> indexInMergedCollEM;
  std::vector<int> indexInMergedCollTRK;
  std::vector<int> indexInMergedCollHAD;
  std::vector<bool> usedSeeds;

  // associating seed to the index of the trackster in the merged collection and the iteration that found it
  std::map<int, std::vector<std::pair<int, TracksterIterIndex>>> seedToTracksterAssociator;
  std::vector<TracksterIterIndex> iterMergedTracksters;
  edm::Handle<std::vector<reco::Track>> track_h;
  evt.getByToken(tracks_token_, track_h);
  const auto &tracks = *track_h;

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  evt.getByToken(clusters_token_, cluster_h);
  const auto &layerClusters = *cluster_h;

  edm::Handle<std::vector<Trackster>> trackstersem_h;
  evt.getByToken(trackstersem_token_, trackstersem_h);
  const auto &trackstersEM = *trackstersem_h;

  edm::Handle<std::vector<Trackster>> tracksterstrkem_h;
  evt.getByToken(tracksterstrkem_token_, tracksterstrkem_h);
  const auto &trackstersTRKEM = *tracksterstrkem_h;

  edm::Handle<std::vector<Trackster>> tracksterstrk_h;
  evt.getByToken(tracksterstrk_token_, tracksterstrk_h);
  const auto &trackstersTRK = *tracksterstrk_h;

  edm::Handle<std::vector<Trackster>> trackstershad_h;
  evt.getByToken(trackstershad_token_, trackstershad_h);
  const auto &trackstersHAD = *trackstershad_h;

  edm::Handle<std::vector<TICLSeedingRegion>> seedingTrk_h;
  evt.getByToken(seedingTrk_token_, seedingTrk_h);
  const auto &seedingTrk = *seedingTrk_h;
  usedSeeds.resize(tracks.size(), false);

  fillTile(tracksterTile, trackstersTRKEM, TracksterIterIndex::TRKEM);
  fillTile(tracksterTile, trackstersEM, TracksterIterIndex::EM);
  fillTile(tracksterTile, trackstersTRK, TracksterIterIndex::TRK);
  fillTile(tracksterTile, trackstersHAD, TracksterIterIndex::HAD);

  auto totalNumberOfTracksters =
      trackstersTRKEM.size() + trackstersTRK.size() + trackstersEM.size() + trackstersHAD.size();
  resultTrackstersMerged->reserve(totalNumberOfTracksters);
  iterMergedTracksters.reserve(totalNumberOfTracksters);
  usedTrackstersMerged.resize(totalNumberOfTracksters, false);
  indexInMergedCollTRKEM.reserve(trackstersTRKEM.size());
  indexInMergedCollEM.reserve(trackstersEM.size());
  indexInMergedCollTRK.reserve(trackstersTRK.size());
  indexInMergedCollHAD.reserve(trackstersHAD.size());

  if (debug_) {
    printTrackstersDebug(trackstersTRKEM, "tracksterTRKEM");
    printTrackstersDebug(trackstersEM, "tracksterEM");
    printTrackstersDebug(trackstersTRK, "tracksterTRK");
    printTrackstersDebug(trackstersHAD, "tracksterHAD");
  }

  for (auto const &t : trackstersTRKEM) {
    indexInMergedCollTRKEM.push_back(resultTrackstersMerged->size());
    seedToTracksterAssociator[t.seedIndex()].emplace_back(resultTrackstersMerged->size(), TracksterIterIndex::TRKEM);
    resultTrackstersMerged->push_back(t);
    iterMergedTracksters.push_back(TracksterIterIndex::TRKEM);
  }

  for (auto const &t : trackstersEM) {
    indexInMergedCollEM.push_back(resultTrackstersMerged->size());
    resultTrackstersMerged->push_back(t);
    iterMergedTracksters.push_back(TracksterIterIndex::EM);
  }

  for (auto const &t : trackstersTRK) {
    indexInMergedCollTRK.push_back(resultTrackstersMerged->size());
    seedToTracksterAssociator[t.seedIndex()].emplace_back(resultTrackstersMerged->size(), TracksterIterIndex::TRK);
    resultTrackstersMerged->push_back(t);
    iterMergedTracksters.push_back(TracksterIterIndex::TRK);
  }

  for (auto const &t : trackstersHAD) {
    indexInMergedCollHAD.push_back(resultTrackstersMerged->size());
    resultTrackstersMerged->push_back(t);
    iterMergedTracksters.push_back(TracksterIterIndex::HAD);
  }

  assignPCAtoTracksters(*resultTrackstersMerged, layerClusters, rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z());
  energyRegressionAndID(layerClusters, *resultTrackstersMerged);

  printTrackstersDebug(*resultTrackstersMerged, "TrackstersMergeProducer");

  auto trackstersMergedHandle = evt.put(std::move(resultTrackstersMerged));

  // TICL Candidate creation
  // We start from neutrals first

  // Photons
  for (unsigned i = 0; i < trackstersEM.size(); ++i) {
    auto mergedIdx = indexInMergedCollEM[i];
    usedTrackstersMerged[mergedIdx] = true;
    const auto &t = trackstersEM[i];  //trackster
    TICLCandidate tmpCandidate;
    tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, mergedIdx));
    tmpCandidate.setCharge(0);
    tmpCandidate.setPdgId(22);
    tmpCandidate.setRawEnergy(t.raw_energy());
    math::XYZTLorentzVector p4(t.raw_energy() * t.barycenter().unit().x(),
                               t.raw_energy() * t.barycenter().unit().y(),
                               t.raw_energy() * t.barycenter().unit().z(),
                               t.raw_energy());
    tmpCandidate.setP4(p4);
    resultCandidates->push_back(tmpCandidate);
  }

  // Neutral Hadrons
  constexpr float mpion2 = 0.13957f * 0.13957f;
  for (unsigned i = 0; i < trackstersHAD.size(); ++i) {
    auto mergedIdx = indexInMergedCollHAD[i];
    usedTrackstersMerged[mergedIdx] = true;
    const auto &t = trackstersHAD[i];  //trackster
    TICLCandidate tmpCandidate;
    tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, mergedIdx));
    tmpCandidate.setCharge(0);
    tmpCandidate.setPdgId(130);
    tmpCandidate.setRawEnergy(t.raw_energy());
    float momentum = std::sqrt(t.raw_energy() * t.raw_energy() - mpion2);
    math::XYZTLorentzVector p4(momentum * t.barycenter().unit().x(),
                               momentum * t.barycenter().unit().y(),
                               momentum * t.barycenter().unit().z(),
                               t.raw_energy());
    tmpCandidate.setP4(p4);
    resultCandidates->push_back(tmpCandidate);
  }

  // Charged Particles
  for (unsigned i = 0; i < trackstersTRKEM.size(); ++i) {
    auto mergedIdx = indexInMergedCollTRKEM[i];
    if (!usedTrackstersMerged[mergedIdx]) {
      const auto &t = trackstersTRKEM[i];  //trackster
      auto trackIdx = t.seedIndex();
      auto const &track = tracks[trackIdx];
      if (!usedSeeds[trackIdx] and t.raw_energy() > 0) {
        usedSeeds[trackIdx] = true;
        usedTrackstersMerged[mergedIdx] = true;

        std::vector<int> trackstersTRKwithSameSeed;
        std::vector<int> trackstersTRKEMwithSameSeed;

        for (const auto &tracksterIterationPair : seedToTracksterAssociator[trackIdx]) {
          if (tracksterIterationPair.first != mergedIdx and !usedTrackstersMerged[tracksterIterationPair.first] and
              trackstersMergedHandle->at(tracksterIterationPair.first).raw_energy() > 0.) {
            if (tracksterIterationPair.second == TracksterIterIndex::TRKEM) {
              trackstersTRKEMwithSameSeed.push_back(tracksterIterationPair.first);
            } else if (tracksterIterationPair.second == TracksterIterIndex::TRK) {
              trackstersTRKwithSameSeed.push_back(tracksterIterationPair.first);
            }
          }
        }

        float tracksterTotalRawPt = t.raw_pt();
        std::vector<int> haloTrackstersTRKIdx;
        bool foundCompatibleTRK = false;

        for (auto otherTracksterIdx : trackstersTRKwithSameSeed) {
          usedTrackstersMerged[otherTracksterIdx] = true;
          tracksterTotalRawPt += trackstersMergedHandle->at(otherTracksterIdx).raw_pt();

          // Check the X,Y,Z barycenter and merge if they are very close (halo)
          if ((t.barycenter() - trackstersMergedHandle->at(otherTracksterIdx).barycenter()).mag2() <
              halo_max_distance2_) {
            haloTrackstersTRKIdx.push_back(otherTracksterIdx);

          } else {
            foundCompatibleTRK = true;
          }
        }

        //check if there is 1-to-1 relationship
        if (trackstersTRKEMwithSameSeed.empty()) {
          if (foundCompatibleTRK) {
            TICLCandidate tmpCandidate;
            tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, mergedIdx));
            double raw_energy = t.raw_energy();

            tmpCandidate.setCharge(track.charge());
            tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, trackIdx));
            tmpCandidate.setPdgId(211 * track.charge());
            for (auto otherTracksterIdx : trackstersTRKwithSameSeed) {
              tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, otherTracksterIdx));
              raw_energy += trackstersMergedHandle->at(otherTracksterIdx).raw_energy();
            }
            tmpCandidate.setRawEnergy(raw_energy);
            math::XYZTLorentzVector p4(raw_energy * track.momentum().unit().x(),
                                       raw_energy * track.momentum().unit().y(),
                                       raw_energy * track.momentum().unit().z(),
                                       raw_energy);
            tmpCandidate.setP4(p4);
            resultCandidates->push_back(tmpCandidate);

          } else {
            TICLCandidate tmpCandidate;
            tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, mergedIdx));
            double raw_energy = t.raw_energy();
            tmpCandidate.setCharge(track.charge());
            tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, trackIdx));
            for (auto otherTracksterIdx : trackstersTRKwithSameSeed) {
              tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, otherTracksterIdx));
              raw_energy += trackstersMergedHandle->at(otherTracksterIdx).raw_energy();
            }
            tmpCandidate.setPdgId(11 * track.charge());

            tmpCandidate.setRawEnergy(raw_energy);
            math::XYZTLorentzVector p4(raw_energy * track.momentum().unit().x(),
                                       raw_energy * track.momentum().unit().y(),
                                       raw_energy * track.momentum().unit().z(),
                                       raw_energy);
            tmpCandidate.setP4(p4);
            resultCandidates->push_back(tmpCandidate);
          }

        } else {
          // if 1-to-many find closest trackster in momentum
          int closestTrackster = mergedIdx;
          float minPtDiff = std::abs(t.raw_pt() - track.pt());
          for (auto otherTracksterIdx : trackstersTRKEMwithSameSeed) {
            auto thisPt = tracksterTotalRawPt + trackstersMergedHandle->at(otherTracksterIdx).raw_pt() - t.raw_pt();
            closestTrackster = std::abs(thisPt - track.pt()) < minPtDiff ? otherTracksterIdx : closestTrackster;
          }
          tracksterTotalRawPt += trackstersMergedHandle->at(closestTrackster).raw_pt() - t.raw_pt();
          usedTrackstersMerged[closestTrackster] = true;

          if (foundCompatibleTRK) {
            TICLCandidate tmpCandidate;
            tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, closestTrackster));
            double raw_energy = trackstersMergedHandle->at(closestTrackster).raw_energy();

            tmpCandidate.setCharge(track.charge());
            tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, trackIdx));
            tmpCandidate.setPdgId(211 * track.charge());
            for (auto otherTracksterIdx : trackstersTRKwithSameSeed) {
              tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, otherTracksterIdx));
              raw_energy += trackstersMergedHandle->at(otherTracksterIdx).raw_energy();
            }
            tmpCandidate.setRawEnergy(raw_energy);
            float momentum = std::sqrt(raw_energy * raw_energy - mpion2);
            math::XYZTLorentzVector p4(momentum * track.momentum().unit().x(),
                                       momentum * track.momentum().unit().y(),
                                       momentum * track.momentum().unit().z(),
                                       raw_energy);
            tmpCandidate.setP4(p4);
            resultCandidates->push_back(tmpCandidate);

          } else {
            TICLCandidate tmpCandidate;
            tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, closestTrackster));
            double raw_energy = trackstersMergedHandle->at(closestTrackster).raw_energy();

            tmpCandidate.setCharge(track.charge());
            tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, trackIdx));
            for (auto otherTracksterIdx : trackstersTRKwithSameSeed) {
              tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, otherTracksterIdx));
              raw_energy += trackstersMergedHandle->at(otherTracksterIdx).raw_energy();
            }
            tmpCandidate.setPdgId(11 * track.charge());
            tmpCandidate.setRawEnergy(raw_energy);
            math::XYZTLorentzVector p4(raw_energy * track.momentum().unit().x(),
                                       raw_energy * track.momentum().unit().y(),
                                       raw_energy * track.momentum().unit().z(),
                                       raw_energy);
            tmpCandidate.setP4(p4);
            resultCandidates->push_back(tmpCandidate);
          }
          // Promote all other TRKEM tracksters as photons with their energy.
          for (auto otherTracksterIdx : trackstersTRKEMwithSameSeed) {
            auto tmpIndex = (otherTracksterIdx != closestTrackster) ? otherTracksterIdx : mergedIdx;
            TICLCandidate photonCandidate;
            const auto &otherTrackster = trackstersMergedHandle->at(tmpIndex);
            auto gammaEnergy = otherTrackster.raw_energy();
            photonCandidate.setCharge(0);
            photonCandidate.setPdgId(22);
            photonCandidate.setRawEnergy(gammaEnergy);
            math::XYZTLorentzVector gammaP4(gammaEnergy * otherTrackster.barycenter().unit().x(),
                                            gammaEnergy * otherTrackster.barycenter().unit().y(),
                                            gammaEnergy * otherTrackster.barycenter().unit().z(),
                                            gammaEnergy);
            photonCandidate.setP4(gammaP4);
            photonCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, tmpIndex));
            resultCandidates->push_back(photonCandidate);
          }
        }
      }
    }
  }  //end of loop over trackstersTRKEM

  for (unsigned i = 0; i < trackstersTRK.size(); ++i) {
    auto mergedIdx = indexInMergedCollTRK[i];
    const auto &t = trackstersTRK[i];  //trackster

    if (!usedTrackstersMerged[mergedIdx] and t.raw_energy() > 0) {
      auto trackIdx = t.seedIndex();
      auto const &track = tracks[trackIdx];
      if (!usedSeeds[trackIdx]) {
        usedSeeds[trackIdx] = true;
        usedTrackstersMerged[mergedIdx] = true;
        TICLCandidate tmpCandidate;
        tmpCandidate.addTrackster(edm::Ptr<ticl::Trackster>(trackstersMergedHandle, mergedIdx));
        tmpCandidate.setCharge(track.charge());
        tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, trackIdx));
        tmpCandidate.setPdgId(211 * track.charge());
        tmpCandidate.setRawEnergy(t.raw_energy());
        float momentum = std::sqrt(t.raw_energy() * t.raw_energy() - mpion2);
        math::XYZTLorentzVector p4(momentum * track.momentum().unit().x(),
                                   momentum * track.momentum().unit().y(),
                                   momentum * track.momentum().unit().z(),
                                   t.raw_energy());
        tmpCandidate.setP4(p4);
        resultCandidates->push_back(tmpCandidate);
      }
    }
  }
  // For all seeds that have 0-energy tracksters whose track is not marked as used, create a charged hadron with the track information.
  for (auto const &s : seedingTrk) {
    if (usedSeeds[s.index] == false) {
      auto const &track = tracks[s.index];
      // emit a charged hadron
      TICLCandidate tmpCandidate;
      tmpCandidate.setCharge(track.charge());
      tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, s.index));
      tmpCandidate.setPdgId(211 * track.charge());
      float energy = std::sqrt(track.pt() * track.pt() + mpion2);
      tmpCandidate.setRawEnergy(energy);
      math::XYZTLorentzVector p4(track.momentum().x(), track.momentum().y(), track.momentum().z(), energy);
      tmpCandidate.setP4(p4);
      resultCandidates->push_back(tmpCandidate);
      usedSeeds[s.index] = true;
    }
  }

  // for all general tracks (high purity, pt > 1), check if they have been used: if not, promote them as charged hadrons
  for (unsigned i = 0; i < tracks.size(); ++i) {
    auto const &track = tracks[i];
    if (track.pt() > track_min_pt_ and track.quality(reco::TrackBase::highPurity) and
        track.missingOuterHits() < track_max_missing_outerhits_ and std::abs(track.outerEta()) > track_min_eta_ and
        std::abs(track.outerEta()) < track_max_eta_ and usedSeeds[i] == false) {
      // emit a charged hadron
      TICLCandidate tmpCandidate;
      tmpCandidate.setCharge(track.charge());
      tmpCandidate.setTrackPtr(edm::Ptr<reco::Track>(track_h, i));
      tmpCandidate.setPdgId(211 * track.charge());
      float energy = std::sqrt(track.pt() * track.pt() + mpion2);
      tmpCandidate.setRawEnergy(energy);
      math::XYZTLorentzVector p4(track.momentum().x(), track.momentum().y(), track.momentum().z(), energy);
      tmpCandidate.setP4(p4);
      resultCandidates->push_back(tmpCandidate);
      usedSeeds[i] = true;
    }
  }

  evt.put(std::move(resultCandidates));
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
  desc.add<edm::InputTag>("tracksterstrkem", edm::InputTag("ticlTrackstersTrkEM"));
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
