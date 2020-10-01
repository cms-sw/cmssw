#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace track = edm::soa::col::pf::track;
namespace rechit = edm::soa::col::pf::rechit;
namespace cluster = edm::soa::col::pf::cluster;

// This class is used to find all links between Tracks and HCAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerTrackHcal : public KDTreeLinkerBase {
public:
  KDTreeLinkerTrackHcal(const edm::ParameterSet& conf);
  ~KDTreeLinkerTrackHcal() override;

  // The KDTree building from rechits list.
  void buildTree(const PFTables& pftables) override;

  // Here we will iterate over all tracks. For each track intersection point with HCAL,
  // we will search the closest rechits in the KDTree, from rechits we will find the
  // hcalClusters and after that we will check the links between the track and
  // all closest hcalClusters.
  void searchLinks(const PFTables& pftables, reco::PFMultiLinksIndex& multilinks) override;

  // Here we free all allocated structures.
  void clear() override;

private:
  // KD trees
  KDTreeLinkerAlgo<size_t> tree_;

  // TrajectoryPoints
  std::string trajectoryLayerEntranceString_;
  std::string trajectoryLayerExitString_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerEntrance_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerExit_;
  bool checkExit_;

  // Hcal-track links
  int nMaxHcalLinksPerTrack_;
};

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, KDTreeLinkerTrackHcal, "KDTreeTrackAndHCALLinker");

KDTreeLinkerTrackHcal::KDTreeLinkerTrackHcal(const edm::ParameterSet& conf)
    : KDTreeLinkerBase(conf),
      trajectoryLayerEntranceString_(conf.getParameter<std::string>("trajectoryLayerEntrance")),
      trajectoryLayerExitString_(conf.getParameter<std::string>("trajectoryLayerExit")),
      nMaxHcalLinksPerTrack_(conf.getParameter<int>("nMaxHcalLinksPerTrack")) {
  // Initialization
  cristalPhiEtaMaxSize_ = 0.2;
  phiOffset_ = 0.32;
  // convert TrajectoryLayers info from string to enum
  trajectoryLayerEntrance_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerEntranceString_);
  trajectoryLayerExit_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerExitString_);
  // make sure the requested setting is supported
  assert((trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
          trajectoryLayerExit_ == reco::PFTrajectoryPoint::HCALExit) ||
         (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
          trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown) ||
         (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::VFcalEntrance &&
          trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown));
  // flag if exit layer should be checked or not
  checkExit_ = trajectoryLayerExit_ != reco::PFTrajectoryPoint::Unknown;
}

KDTreeLinkerTrackHcal::~KDTreeLinkerTrackHcal() { clear(); }

void KDTreeLinkerTrackHcal::buildTree(const PFTables& pftables) {
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo<size_t, 2>> eltList;
  const auto& clusters = pftables.getClusterTable(_fieldType);
  const auto& rechitTable = clusters.rechit_table;

  // Filling of this list
  for (size_t irechit = 0; irechit < rechitTable.size(); irechit++) {
    KDTreeNodeInfo<size_t, 2> rh1(
        irechit, float(rechitTable.get<rechit::Eta>(irechit)), float(rechitTable.get<rechit::Phi>(irechit)));
    eltList.push_back(rh1);

    // Here we solve the problem of phi circular set by duplicating some rechits
    // too close to -Pi (or to Pi) and adding (substracting) to them 2 * Pi.
    if (rh1.dims[1] > (M_PI - phiOffset_)) {
      float phi = rh1.dims[1] - 2 * M_PI;
      KDTreeNodeInfo<size_t, 2> rh2(irechit, float(rechitTable.get<rechit::Eta>(irechit)), phi);
      eltList.push_back(rh2);
    }

    if (rh1.dims[1] < (M_PI * -1.0 + phiOffset_)) {
      float phi = rh1.dims[1] + 2 * M_PI;
      KDTreeNodeInfo<size_t, 2> rh3(irechit, float(rechitTable.get<rechit::Eta>(irechit)), phi);
      eltList.push_back(rh3);
    }
  }

  // Here we define the upper/lower bounds of the 2D space (eta/phi).
  float phimin = -1.0 * M_PI - phiOffset_;
  float phimax = M_PI + phiOffset_;

  // etamin-etamax, phimin-phimax
  KDTreeBox region(-3.0f, 3.0f, phimin, phimax);

  // We may now build the KDTree
  tree_.build(eltList, region);
}

void KDTreeLinkerTrackHcal::searchLinks(const PFTables& pftables, reco::PFMultiLinksIndex& multilinks) {
  // Most of the code has been taken from LinkByRecHit.cc
  LogDebug("KDTreeLinkerTrackHcal") << "searchLinks fieldType=" << _fieldType << " targetType=" << _targetType;

  const auto& trackTableEntrance = pftables.getTrackTable(trajectoryLayerEntrance_);
  const auto& trackTableExit = checkExit_ ? pftables.getTrackTable(trajectoryLayerExit_) : trackTableEntrance;
  const auto& clusters = pftables.getClusterTable(_fieldType);
  const auto& clusterTable = clusters.cluster_table;
  const auto& rechitTable = clusters.rechit_table;

  std::unordered_map<size_t, std::vector<size_t>> track_to_cluster_all;

  // We iterate over the tracks.
  LogDebug("KDTreeLinkerTrackHcal") << "looping over " << trackTableEntrance.size() << " tracks";
  for (size_t itrack = 0; itrack < trackTableEntrance.size(); itrack++) {
    if (not trackTableEntrance.get<track::ExtrapolationValid>(itrack)) {
      continue;
    }

    const double atEntranceEta = trackTableEntrance.get<track::Eta>(itrack);
    const float atEntrancePhi = trackTableEntrance.get<track::Phi>(itrack);

    // In case the exit point check is requested, check eta and phi differences between entrance and exit
    double dHeta = 0.0;
    float dHphi = 0.0;
    if (checkExit_) {
      const double atExitEta = trackTableExit.get<track::Eta>(itrack);
      const float atExitPhi = trackTableExit.get<track::Phi>(itrack);
      dHeta = atExitEta - atEntranceEta;
      dHphi = atExitPhi - atEntrancePhi;
      if (dHphi > M_PI)
        dHphi = dHphi - 2. * M_PI;
      else if (dHphi < -M_PI)
        dHphi = dHphi + 2. * M_PI;
    }  // checkExit_

    float tracketa = atEntranceEta + 0.1 * dHeta;
    float trackphi = atEntrancePhi + 0.1 * dHphi;

    if (trackphi > M_PI)
      trackphi -= 2 * M_PI;
    else if (trackphi < -M_PI)
      trackphi += 2 * M_PI;

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double inflation = 1.;
    float rangeeta = (cristalPhiEtaMaxSize_ * (1.5 + 0.5) + 0.2 * std::abs(dHeta)) * inflation;
    float rangephi = (cristalPhiEtaMaxSize_ * (1.5 + 0.5) + 0.2 * std::abs(dHphi)) * inflation;

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<size_t> recHits;
    KDTreeBox trackBox(tracketa - rangeeta, tracketa + rangeeta, trackphi - rangephi, trackphi + rangephi);
    tree_.search(trackBox, recHits);
    LogDebug("KDTreeLinkerTrackHcal") << "found " << recHits.size() << " associated to track " << itrack;

    // Here we check all rechit candidates using the non-approximated method.
    for (const size_t irechit : recHits) {
      const auto& corner_eta = rechitTable.get<rechit::CornerEta>(irechit);
      const auto& corner_phi = rechitTable.get<rechit::CornerPhi>(irechit);

      double rhsizeeta = std::abs(corner_eta[3] - corner_eta[1]);
      double rhsizephi = std::abs(reco::deltaPhi(corner_phi[3], corner_phi[1]));

      double deta = std::abs(rechitTable.get<rechit::Eta>(irechit) - tracketa);
      double dphi = std::abs(reco::deltaPhi(rechitTable.get<rechit::Phi>(irechit), trackphi));

      // Find all clusters associated to given rechit
      const auto& rechit_clusters = clusters.rechit_to_cluster.at(irechit);
      LogDebug("KDTreeLinkerTrackHcal") << "rechit " << irechit << " has " << rechit_clusters.size() << " clusters";

      for (const size_t clusteridx : rechit_clusters) {
        int fracsNbr = clusterTable.get<cluster::FracsNbr>(clusteridx);
        double _rhsizeeta = rhsizeeta * (1.5 + 0.5 / fracsNbr) + 0.2 * std::abs(dHeta);
        double _rhsizephi = rhsizephi * (1.5 + 0.5 / fracsNbr) + 0.2 * std::abs(dHphi);

        // Check if the track and the cluster are linked
        if (deta < (_rhsizeeta / 2.) && dphi < (_rhsizephi / 2.)) {
          if (nMaxHcalLinksPerTrack_ < 0 || pftables.track_table_vertex.get<track::IsLinkedToDisplacedVertex>(itrack)) {
            multilinks.addLink(
                pftables.track_to_element[itrack], clusters.cluster_to_element[clusteridx], _targetType, _fieldType);
          } else {
            LogDebug("KDTreeLinkerTrackHcal")
                << "tentatively storing link clusteridx=" << clusteridx << " trackidx=" << itrack;
            track_to_cluster_all[itrack].push_back(clusteridx);
          }
        }
      }  //loop over clusters
    }    //loop over rechits
  }      //loop over tracks

  LogDebug("KDTreeLinkerTrackHcal") << "finalizing links, track_to_cluster_all=" << track_to_cluster_all.size();

  for (const auto& track_to_cluster : track_to_cluster_all) {
    size_t trackidx = track_to_cluster.first;
    const auto& clusterindexes = track_to_cluster.second;
    std::vector<double> distances;
    distances.reserve(clusterindexes.size());

    for (const auto& clusteridx : clusterindexes) {
      const double dist = LinkByRecHit::computeTrackHCALDist(
          checkExit_, trackidx, clusteridx, clusterTable, trackTableEntrance, trackTableExit);
      distances.push_back(dist);
    }

    for (auto i : sort_indexes(distances)) {
      size_t clusteridx = clusterindexes[i];

      multilinks.addLink(
          pftables.track_to_element[trackidx], clusters.cluster_to_element[clusteridx], _targetType, _fieldType);

      if (multilinks.getNumLinks(pftables.track_to_element[trackidx], _targetType, _fieldType) >=
          (unsigned)nMaxHcalLinksPerTrack_) {
        LogDebug("KDTreeLinkerTrackHcal") << "reached max links for trackidx=" << trackidx;
        break;
      }
    }
  }
  LogDebug("KDTreeLinkerTrackHcal") << "searchLinks done";
}

void KDTreeLinkerTrackHcal::clear() { tree_.clear(); }
