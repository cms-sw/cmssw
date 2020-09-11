#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// This class is used to find all links between Tracks and HCAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerTrackHcal : public KDTreeLinkerBase {
public:
  KDTreeLinkerTrackHcal(const edm::ParameterSet& conf);
  ~KDTreeLinkerTrackHcal() override;

  // With this method, we create the list of track that we want to link.
  void insertTargetElt(reco::PFBlockElement* track) override;

  // Here, we create the list of hcalCluster that we want to link. From hcalCluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  void insertFieldClusterElt(reco::PFBlockElement* hcalCluster) override;

  // The KDTree building from rechits list.
  void buildTree() override;

  // Here we will iterate over all tracks. For each track intersection point with HCAL,
  // we will search the closest rechits in the KDTree, from rechits we will find the
  // hcalClusters and after that we will check the links between the track and
  // all closest hcalClusters.
  void searchLinks() override;

  // Here, we will store all Track/HCAL founded links in the PFBlockElement class
  // of each psCluster in the PFmultilinks field.
  void updatePFBlockEltWithLinks() override;

  // Here we free all allocated structures.
  void clear() override;

private:
  // Data used by the KDTree algorithm : sets of Tracks and HCAL clusters.
  BlockEltSet targetSet_;
  BlockEltSet fieldClusterSet_;

  // Sets of rechits that compose the HCAL clusters.
  RecHitSet rechitsSet_;

  // Map of linked Track/HCAL clusters.
  BlockElt2BlockEltMap target2ClusterLinks_;

  // Map of the HCAL clusters associated to a rechit.
  RecHit2BlockEltMap rechit2ClusterLinks_;

  // KD trees
  KDTreeLinkerAlgo<reco::PFRecHit const*> tree_;

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

void KDTreeLinkerTrackHcal::insertTargetElt(reco::PFBlockElement* track) {
  if (track->trackRefPF()->extrapolatedPoint(trajectoryLayerEntrance_).isValid()) {
    targetSet_.insert(track);
  }
}

void KDTreeLinkerTrackHcal::insertFieldClusterElt(reco::PFBlockElement* hcalCluster) {
  const reco::PFClusterRef& clusterref = hcalCluster->clusterRef();

  // This test is more or less done in PFBlockAlgo.h. In others cases, it should be switch on.
  //   if (!((clusterref->layer() == PFLayer::HCAL_ENDCAP) ||
  // 	(clusterref->layer() == PFLayer::HCAL_BARREL1)))
  //     return;

  const std::vector<reco::PFRecHitFraction>& fraction = clusterref->recHitFractions();

  // We create a list of hcalCluster
  fieldClusterSet_.insert(hcalCluster);
  for (size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < cutOffFrac))
      continue;

    const reco::PFRecHit& rechit = *rh;

    // We save the links rechit to HcalClusters
    rechit2ClusterLinks_[&rechit].insert(hcalCluster);

    // We create a liste of rechits
    rechitsSet_.insert(&rechit);
  }
}

void KDTreeLinkerTrackHcal::buildTree() {
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo<reco::PFRecHit const*, 2>> eltList;

  // Filling of this list
  for (RecHitSet::const_iterator it = rechitsSet_.begin(); it != rechitsSet_.end(); it++) {
    const reco::PFRecHit::REPPoint& posrep = (*it)->positionREP();

    KDTreeNodeInfo<reco::PFRecHit const*, 2> rh1(*it, posrep.eta(), posrep.phi());
    eltList.push_back(rh1);

    // Here we solve the problem of phi circular set by duplicating some rechits
    // too close to -Pi (or to Pi) and adding (substracting) to them 2 * Pi.
    if (rh1.dims[1] > (M_PI - phiOffset_)) {
      float phi = rh1.dims[1] - 2 * M_PI;
      KDTreeNodeInfo<reco::PFRecHit const*, 2> rh2(*it, float(posrep.eta()), phi);
      eltList.push_back(rh2);
    }

    if (rh1.dims[1] < (M_PI * -1.0 + phiOffset_)) {
      float phi = rh1.dims[1] + 2 * M_PI;
      KDTreeNodeInfo<reco::PFRecHit const*, 2> rh3(*it, float(posrep.eta()), phi);
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

void KDTreeLinkerTrackHcal::searchLinks() {
  // Most of the code has been taken from LinkByRecHit.cc

  // We iterate over the tracks.
  for (BlockEltSet::iterator it = targetSet_.begin(); it != targetSet_.end(); it++) {
    reco::PFRecTrackRef trackref = (*it)->trackRefPF();

    const reco::PFTrajectoryPoint& atHCAL = trackref->extrapolatedPoint(trajectoryLayerEntrance_);

    // The track didn't reach hcal
    if (!atHCAL.isValid())
      continue;

    // In case the exit point check is requested, check eta and phi differences between entrance and exit
    double dHeta = 0.0;
    float dHphi = 0.0;
    if (checkExit_) {
      const reco::PFTrajectoryPoint& atHCALExit = trackref->extrapolatedPoint(trajectoryLayerExit_);
      dHeta = atHCALExit.positionREP().eta() - atHCAL.positionREP().eta();
      dHphi = atHCALExit.positionREP().phi() - atHCAL.positionREP().phi();
      if (dHphi > M_PI)
        dHphi = dHphi - 2. * M_PI;
      else if (dHphi < -M_PI)
        dHphi = dHphi + 2. * M_PI;
    }  // checkExit_

    float tracketa = atHCAL.positionREP().eta() + 0.1 * dHeta;
    float trackphi = atHCAL.positionREP().phi() + 0.1 * dHphi;

    if (trackphi > M_PI)
      trackphi -= 2 * M_PI;
    else if (trackphi < -M_PI)
      trackphi += 2 * M_PI;

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double inflation = 1.;
    float rangeeta = (cristalPhiEtaMaxSize_ * (1.5 + 0.5) + 0.2 * fabs(dHeta)) * inflation;
    float rangephi = (cristalPhiEtaMaxSize_ * (1.5 + 0.5) + 0.2 * fabs(dHphi)) * inflation;

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<reco::PFRecHit const*> recHits;
    KDTreeBox trackBox(tracketa - rangeeta, tracketa + rangeeta, trackphi - rangephi, trackphi + rangephi);
    tree_.search(trackBox, recHits);

    // Here we check all rechit candidates using the non-approximated method.
    for (auto const& recHit : recHits) {
      const auto& rhrep = recHit->positionREP();
      const auto& corners = recHit->getCornersREP();

      double rhsizeeta = fabs(corners[3].eta() - corners[1].eta());
      double rhsizephi = fabs(corners[3].phi() - corners[1].phi());
      if (rhsizephi > M_PI)
        rhsizephi = 2. * M_PI - rhsizephi;

      double deta = fabs(rhrep.eta() - tracketa);
      double dphi = fabs(rhrep.phi() - trackphi);
      if (dphi > M_PI)
        dphi = 2. * M_PI - dphi;

      // Find all clusters associated to given rechit
      RecHit2BlockEltMap::iterator ret = rechit2ClusterLinks_.find(recHit);

      for (BlockEltSet::iterator clusterIt = ret->second.begin(); clusterIt != ret->second.end(); clusterIt++) {
        const reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
        int fracsNbr = clusterref->recHitFractions().size();

        double _rhsizeeta = rhsizeeta * (1.5 + 0.5 / fracsNbr) + 0.2 * fabs(dHeta);
        double _rhsizephi = rhsizephi * (1.5 + 0.5 / fracsNbr) + 0.2 * fabs(dHphi);

        // Check if the track and the cluster are linked
        if (deta < (_rhsizeeta / 2.) && dphi < (_rhsizephi / 2.))
          target2ClusterLinks_[*it].insert(*clusterIt);
      }
    }
  }
}

void KDTreeLinkerTrackHcal::updatePFBlockEltWithLinks() {
  //TODO YG : Check if cluster positionREP() is valid ?

  // Here we save in each track the list of phi/eta values of linked clusters.
  for (BlockElt2BlockEltMap::iterator it = target2ClusterLinks_.begin(); it != target2ClusterLinks_.end(); ++it) {
    const auto& trackElt = it->first;
    const auto& hcalEltSet = it->second;
    reco::PFMultiLinksTC multitracks(true);

    //
    // No restriction on the number of HCAL links per track or isLinkedToDisplacedVertex
    if (nMaxHcalLinksPerTrack_ < 0. || trackElt->isLinkedToDisplacedVertex()) {
      for (const auto& hcalElt : hcalEltSet) {
        double clusterphi = hcalElt->clusterRef()->positionREP().phi();
        double clustereta = hcalElt->clusterRef()->positionREP().eta();
        multitracks.linkedClusters.push_back(std::make_pair(clusterphi, clustereta));

        // We set the multilinks flag of the track (for links to ECAL) to true. It will allow us to
        // use it in an optimized way in prefilter
        hcalElt->setIsValidMultilinks(true, _targetType);
      }

    }
    //
    // Store only the N closest HCAL links per track.
    else {
      const reco::PFRecTrackRef& trackref = trackElt->trackRefPF();
      const reco::PFTrajectoryPoint& tkAtHCALEnt = trackref->extrapolatedPoint(trajectoryLayerEntrance_);
      const reco::PFCluster::REPPoint& tkreppos = tkAtHCALEnt.positionREP();
      // Check exit point
      double dHEta = 0.;
      double dHPhi = 0.;
      double dRHCALEx = 0.;
      if (checkExit_) {
        const reco::PFTrajectoryPoint& tkAtHCALEx = trackref->extrapolatedPoint(trajectoryLayerExit_);
        dHEta = (tkAtHCALEx.positionREP().Eta() - tkAtHCALEnt.positionREP().Eta());
        dHPhi = reco::deltaPhi(tkAtHCALEx.positionREP().Phi(), tkAtHCALEnt.positionREP().Phi());
        dRHCALEx = tkAtHCALEx.position().R();
      }

      std::vector<double> vDist;
      double dist(-1.0);

      // Fill the vector of distances between HCAL clusters and the track
      for (const auto& hcalElt : hcalEltSet) {
        double clusterphi = hcalElt->clusterRef()->positionREP().phi();
        double clustereta = hcalElt->clusterRef()->positionREP().eta();

        // when checkExit_ is false
        if (!checkExit_) {
          dist = LinkByRecHit::computeDist(clustereta, clusterphi, tkreppos.Eta(), tkreppos.Phi());
        }
        // when checkExit_ is true
        else {
          //special case ! A looper  can exit the barrel inwards and hit the endcap
          //In this case calculate the distance based on the first crossing since
          //the looper will probably never make it to the endcap
          if (dRHCALEx < tkAtHCALEnt.position().R()) {
            dist = LinkByRecHit::computeDist(clustereta, clusterphi, tkreppos.Eta(), tkreppos.Phi());
            edm::LogWarning("KDTreeLinkerTrackHcal ")
                << "Special case of linking with track hitting HCAL and looping back in the tracker ";
          } else {
            dist = LinkByRecHit::computeDist(
                clustereta, clusterphi, tkreppos.Eta() + 0.1 * dHEta, tkreppos.Phi() + 0.1 * dHPhi);
          }
        }  // checkExit_

        vDist.push_back(dist);
      }  // loop over hcalEltSet

      // Fill multitracks
      for (auto i : sort_indexes(vDist)) {
        const BlockEltSet::iterator hcalEltIt = std::next(hcalEltSet.begin(), i);
        double clusterphi = (*hcalEltIt)->clusterRef()->positionREP().phi();
        double clustereta = (*hcalEltIt)->clusterRef()->positionREP().eta();
        multitracks.linkedClusters.push_back(std::make_pair(clusterphi, clustereta));

        // We set the multilinks flag of the track (for links to ECAL) to true. It will allow us to
        // use it in an optimized way in prefilter
        (*hcalEltIt)->setIsValidMultilinks(true, _targetType);

        if (multitracks.linkedClusters.size() >= (unsigned)nMaxHcalLinksPerTrack_)
          break;
      }
    }

    // Store multitracks
    trackElt->setMultilinks(multitracks, _fieldType);
  }  // loop over target2ClusterLinks_
}

void KDTreeLinkerTrackHcal::clear() {
  targetSet_.clear();
  fieldClusterSet_.clear();

  rechitsSet_.clear();

  rechit2ClusterLinks_.clear();
  target2ClusterLinks_.clear();

  tree_.clear();
}
