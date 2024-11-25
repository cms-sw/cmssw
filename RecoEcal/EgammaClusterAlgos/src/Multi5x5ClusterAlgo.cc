
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5ClusterAlgo.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterEtLess.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//temporary sorter, I'm sure this must exist already in CMSSW
template <class T1, class T2, typename Comp = std::less<T1> >
struct PairSortByFirst {
  Comp comp;
  bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) { return comp(lhs.first, rhs.first); }
  bool operator()(const T1& lhs, const std::pair<T1, T2>& rhs) { return comp(lhs, rhs.first); }
  bool operator()(const std::pair<T1, T2>& lhs, const T1& rhs) { return comp(lhs.first, rhs); }
  bool operator()(const T1& lhs, const T1& rhs) { return comp(lhs, rhs); }
};

struct EBDetIdSorter {
  bool operator()(const EBDetId& lhs, const EBDetId& rhs) {
    if (lhs.ieta() < rhs.ieta())
      return true;
    else if (lhs.ieta() > rhs.ieta())
      return false;
    else
      return lhs.iphi() < rhs.iphi();
  }
};

struct EEDetIdSorter {
  bool operator()(const EEDetId& lhs, const EEDetId& rhs) {
    if (lhs.zside() < rhs.zside())
      return true;
    else if (lhs.zside() > rhs.zside())
      return false;
    else {  //z is equal, onto ix
      if (lhs.ix() < rhs.ix())
        return true;
      else if (lhs.ix() > rhs.ix())
        return false;
      else
        return lhs.iy() < rhs.iy();
    }
  }
};

// Return a vector of clusters from a collection of EcalRecHits:
//
std::vector<reco::BasicCluster> Multi5x5ClusterAlgo::makeClusters(const EcalRecHitCollection* hits,
                                                                  const CaloSubdetectorGeometry* geometry_p,
                                                                  const CaloSubdetectorTopology* topology_p,
                                                                  const CaloSubdetectorGeometry* geometryES_p,
                                                                  reco::CaloID::Detectors detector,
                                                                  bool regional,
                                                                  const std::vector<RectangularEtaPhiRegion>& regions) {
  std::vector<EcalRecHit> seeds;

  double threshold = 0;
  std::string ecalPart_string;
  detector_ = reco::CaloID::DET_NONE;
  if (detector == reco::CaloID::DET_ECAL_ENDCAP) {
    detector_ = reco::CaloID::DET_ECAL_ENDCAP;
    threshold = ecalEndcapSeedThreshold_;
    ecalPart_string = "EndCap";
  }
  if (detector == reco::CaloID::DET_ECAL_BARREL) {
    detector_ = reco::CaloID::DET_ECAL_BARREL;
    threshold = ecalBarrelSeedThreshold_;
    ecalPart_string = "Barrel";
  }

  LogTrace("EcalClusters") << "-------------------------------------------------------------";
  LogTrace("EcalClusters") << "Island algorithm invoked for ECAL" << ecalPart_string;
  LogTrace("EcalClusters") << "Looking for seeds, energy threshold used = " << threshold << " GeV";

  int nregions = 0;
  if (regional)
    nregions = regions.size();

  if (!regional || nregions) {
    for (auto const& hit : *hits) {
      double energy = hit.energy();
      if (energy < threshold)
        continue;  // need to check to see if this line is useful!

      auto thisCell = geometry_p->getGeometry(hit.id());
      // Require that RecHit is within clustering region in case
      // of regional reconstruction
      bool withinRegion = false;
      if (regional) {
        for (auto const& region : regions) {
          if (region.inRegion(thisCell->etaPos(), thisCell->phiPos())) {
            withinRegion = true;
            break;
          }
        }
      }

      if (!regional || withinRegion) {
        float ET = hit.energy() * thisCell->getPosition().basicVector().unit().perp();
        if (ET > threshold)
          seeds.push_back(hit);
      }
    }
  }

  sort(seeds.begin(), seeds.end(), [](auto const& x, auto const& y) { return (x.energy() > y.energy()); });

  LogTrace("EcalClusters") << "Total number of seeds found in event = " << seeds.size();

  auto clusters_v = mainSearch(hits, geometry_p, topology_p, geometryES_p, seeds);
  sort(clusters_v.rbegin(), clusters_v.rend(), isClusterEtLess);

  LogTrace("EcalClusters") << "---------- end of main search. clusters have been sorted ----";

  return clusters_v;
}

// Search for clusters
//
namespace {
  class ProtoClusters {
    using ProtoBasicCluster = Multi5x5ClusterAlgo::ProtoBasicCluster;
    std::vector<ProtoBasicCluster> clusters_;
    std::vector<std::pair<DetId, int> > whichClusCrysBelongsTo_;
    bool reassignSeedCrysToClusterItSeeds_;

  public:
    ProtoClusters(bool reassignSeedCrysToClusterItSeeds)
        : reassignSeedCrysToClusterItSeeds_(reassignSeedCrysToClusterItSeeds) {}
    void push_back(ProtoBasicCluster&& c) {
      clusters_.push_back(std::move(c));
      if (reassignSeedCrysToClusterItSeeds_) {
        for (auto const& hit : clusters_.back().hits()) {
          whichClusCrysBelongsTo_.push_back(std::pair<DetId, int>(hit.first, clusters_.size()));
        }
      }
    }
    std::vector<ProtoBasicCluster> finalize() {
      if (reassignSeedCrysToClusterItSeeds_) {
        std::sort(whichClusCrysBelongsTo_.begin(), whichClusCrysBelongsTo_.end(), PairSortByFirst<DetId, int>());

        for (size_t clusNr = 0; clusNr < clusters_.size(); clusNr++) {
          if (!clusters_[clusNr].containsSeed()) {
            const EcalRecHit& seedHit = clusters_[clusNr].seed();
            typedef std::vector<std::pair<DetId, int> >::iterator It;
            std::pair<It, It> result = std::equal_range(whichClusCrysBelongsTo_.begin(),
                                                        whichClusCrysBelongsTo_.end(),
                                                        seedHit.id(),
                                                        PairSortByFirst<DetId, int>());

            if (result.first != result.second)
              clusters_[result.first->second].removeHit(seedHit);
            clusters_[clusNr].addSeed();
          }
        }
      }
      whichClusCrysBelongsTo_.clear();
      return std::move(clusters_);
    }
  };
}  // namespace

std::vector<reco::BasicCluster> Multi5x5ClusterAlgo::mainSearch(const EcalRecHitCollection* hits,
                                                                const CaloSubdetectorGeometry* geometry_p,
                                                                const CaloSubdetectorTopology* topology_p,
                                                                const CaloSubdetectorGeometry* geometryES_p,
                                                                std::vector<EcalRecHit> const& seeds) {
  LogTrace("EcalClusters") << "Building clusters............";

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters_v;

  // The set of used DetID's
  std::set<DetId> used_s;
  // set of crystals not to be added but which can seed
  // a new 3x3 (e.g. the outer crystals in a 5x5)
  std::set<DetId> canSeed_s;

  ProtoClusters protoClusters{reassignSeedCrysToClusterItSeeds_};

  // Loop over seeds:
  bool first = true;
  for (auto const& seed : seeds) {
    struct Guard {
      bool& b_;
      Guard(bool& b) : b_{b} {};
      ~Guard() { b_ = false; }
    };
    Guard guard(first);
    // check if this crystal is able to seed
    // (event though it is already used)
    bool usedButCanSeed = false;
    if (canSeed_s.find(seed.id()) != canSeed_s.end())
      usedButCanSeed = true;

    // avoid seeding for anomalous channels
    if (!seed.checkFlag(EcalRecHit::kGood)) {  // if rechit is good, no need for further checks
      if (seed.checkFlags(v_chstatus_))
        continue;
    }

    // make sure the current seed does not belong to a cluster already.
    if ((used_s.find(seed.id()) != used_s.end()) && (usedButCanSeed == false)) {
      if (first) {
        LogTrace("EcalClusters") << "##############################################################";
        LogTrace("EcalClusters") << "DEBUG ALERT: Highest energy seed already belongs to a cluster!";
        LogTrace("EcalClusters") << "##############################################################";
      }

      // seed crystal is used or is used and cannot seed a cluster
      // so continue to the next seed crystal...
      continue;
    }

    // Create a navigator at the seed and get seed
    // energy
    CaloNavigator<DetId> navigator(seed.id(), topology_p);
    DetId seedId = navigator.pos();
    EcalRecHitCollection::const_iterator seedIt = hits->find(seedId);
    navigator.setHome(seedId);

    // Is the seed a local maximum?
    bool localMaxima = checkMaxima(navigator, hits);

    if (localMaxima) {
      // build the 5x5 taking care over which crystals
      // can seed new clusters and which can't
      auto current_v = prepareCluster(navigator, hits, geometry_p, used_s, canSeed_s);

      // If some crystals in the current vector then
      // make them into a cluster
      if (!current_v.empty()) {
        auto c = makeCluster(hits, geometry_p, geometryES_p, seedIt, usedButCanSeed, current_v);
        if (c) {
          protoClusters.push_back(std::move(*c));
        } else {
          for (auto const& current : current_v) {
            used_s.erase(current.first);
          }
        }
      }
    }
  }  // End loop on seed crystals

  auto finalProtoClusters = protoClusters.finalize();

  for (auto const& protoCluster : finalProtoClusters) {
    Point position = posCalculator_.Calculate_Location(protoCluster.hits(), hits, geometry_p, geometryES_p);
    clusters_v.emplace_back(protoCluster.energy(),
                            position,
                            reco::CaloID(detector_),
                            protoCluster.hits(),
                            reco::CaloCluster::multi5x5,
                            protoCluster.seed().id());
  }

  return clusters_v;
}

std::optional<Multi5x5ClusterAlgo::ProtoBasicCluster> Multi5x5ClusterAlgo::makeCluster(
    const EcalRecHitCollection* hits,
    const CaloSubdetectorGeometry* geometry,
    const CaloSubdetectorGeometry* geometryES,
    const EcalRecHitCollection::const_iterator& seedIt,
    bool seedOutside,
    std::vector<std::pair<DetId, float> >& current_v) {
  double energy = 0;
  //double chi2   = 0;
  reco::CaloID caloID;
  Point position;
  position = posCalculator_.Calculate_Location(current_v, hits, geometry, geometryES);

  for (auto const& hitInfo : current_v) {
    EcalRecHitCollection::const_iterator itt = hits->find(hitInfo.first);
    EcalRecHit hit_p = *itt;
    energy += hit_p.energy();
    //chi2 += 0;
    if (hitInfo.first.subdetId() == EcalBarrel) {
      caloID = reco::CaloID::DET_ECAL_BARREL;
    } else {
      caloID = reco::CaloID::DET_ECAL_ENDCAP;
    }
  }
  //chi2 /= energy;

  LogTrace("EcalClusters") << "******** NEW CLUSTER ********";
  LogTrace("EcalClusters") << "No. of crystals = " << current_v.size();
  LogTrace("EcalClusters") << "     Energy     = " << energy;
  LogTrace("EcalClusters") << "     Phi        = " << position.phi();
  LogTrace("EcalClusters") << "     Eta " << position.eta();
  LogTrace("EcalClusters") << "*****************************";

  // to be a valid cluster the cluster energy
  // must be at least the seed energy
  double seedEnergy = seedIt->energy();
  if ((seedOutside && energy >= 0) || (!seedOutside && energy >= seedEnergy)) {
    return ProtoBasicCluster(energy, *seedIt, std::move(current_v));

    // clusters_v.push_back(reco::BasicCluster(energy, position, reco::CaloID(detector_), current_v, reco::CaloCluster::multi5x5, seedIt->id()));

    // if no valid cluster was built,
    // then free up these crystals to be used in the next...
  }  //else
  return {};
}

bool Multi5x5ClusterAlgo::checkMaxima(CaloNavigator<DetId>& navigator, const EcalRecHitCollection* hits) const {
  bool maxima = true;
  EcalRecHitCollection::const_iterator thisHit;
  EcalRecHitCollection::const_iterator seedHit = hits->find(navigator.pos());
  double seedEnergy = seedHit->energy();

  std::array<DetId, 4> swissCrossVec;

  swissCrossVec[0] = navigator.west();
  navigator.home();
  swissCrossVec[1] = navigator.east();
  navigator.home();
  swissCrossVec[2] = navigator.north();
  navigator.home();
  swissCrossVec[3] = navigator.south();
  navigator.home();

  for (auto const& det : swissCrossVec) {
    // look for this hit
    thisHit = hits->find(det);

    // continue if this hit was not found
    if ((det == DetId(0)) || thisHit == hits->end())
      continue;

    // the recHit has to be skipped in the local maximum search if it was found
    // in the map of channels to be excluded
    uint32_t rhFlag = thisHit->recoFlag();
    std::vector<int>::const_iterator vit = std::find(v_chstatus_.begin(), v_chstatus_.end(), rhFlag);
    if (vit != v_chstatus_.end())
      continue;

    // if this crystal has more energy than the seed then we do
    // not have a local maxima
    if (thisHit->energy() > seedEnergy) {
      maxima = false;
      break;
    }
  }

  return maxima;
}

std::vector<std::pair<DetId, float> > Multi5x5ClusterAlgo::prepareCluster(CaloNavigator<DetId>& navigator,
                                                                          const EcalRecHitCollection* hits,
                                                                          const CaloSubdetectorGeometry* geometry,
                                                                          std::set<DetId>& used_s,
                                                                          std::set<DetId>& canSeed_s) const {
  std::vector<std::pair<DetId, float> > current_v;
  DetId thisDet;
  std::set<DetId>::iterator setItr;

  // now add the 5x5 taking care to mark the edges
  // as able to seed and where overlapping in the central
  // region with crystals that were previously able to seed
  // change their status so they are not able to seed
  //std::cout << std::endl;
  for (int dx = -2; dx < 3; ++dx) {
    for (int dy = -2; dy < 3; ++dy) {
      // navigate in free steps forming
      // a full 5x5
      thisDet = navigator.offsetBy(dx, dy);
      navigator.home();

      // add the current crystal
      //std::cout << "adding " << dx << ", " << dy << std::endl;
      if (addCrystal(thisDet, *hits) and (used_s.find(thisDet) == used_s.end())) {
        current_v.push_back(std::pair<DetId, float>(thisDet, 1.));  // by default hit energy fractions are set at 1.
        used_s.insert(thisDet);
      }

      // now consider if we are in an edge (outer 16)
      // or central (inner 9) region
      if ((abs(dx) > 1) || (abs(dy) > 1)) {
        // this is an "edge" so should be allowed to seed
        // provided it is not already used
        //std::cout << "   setting can seed" << std::endl;
        canSeed_s.insert(thisDet);
      }  // end if "edge"
      else {
        // or else we are in the central 3x3
        // and must remove any of these crystals from the canSeed set
        setItr = canSeed_s.find(thisDet);
        if (setItr != canSeed_s.end()) {
          //std::cout << "   unsetting can seed" << std::endl;
          canSeed_s.erase(setItr);
        }
      }  // end if "centre"

    }  // end loop on dy

  }  // end loop on dx

  //std::cout << "*** " << std::endl;
  //std::cout << " current_v contains " << current_v.size() << std::endl;
  //std::cout << "*** " << std::endl;
  return current_v;
}

bool Multi5x5ClusterAlgo::addCrystal(const DetId& det, const EcalRecHitCollection& recHits) {
  auto thisIt = recHits.find(det);
  return ((thisIt != recHits.end()) && (thisIt->id() != DetId(0)));
}

bool Multi5x5ClusterAlgo::ProtoBasicCluster::removeHit(const EcalRecHit& hitToRM) {
  std::vector<std::pair<DetId, float> >::iterator hitPos;
  for (hitPos = hits_.begin(); hitPos < hits_.end(); hitPos++) {
    if (hitToRM.id() == hitPos->first)
      break;
  }
  if (hitPos != hits_.end()) {
    hits_.erase(hitPos);
    energy_ -= hitToRM.energy();
    return true;
  }
  return false;
}

//now the tricky thing is to make sure we insert it in the right place in the hit vector
//order is from -2,-2, -2,-1 to 2,2 with seed being 0,0 (eta,phi)
bool Multi5x5ClusterAlgo::ProtoBasicCluster::addSeed() {
  typedef std::vector<std::pair<DetId, float> >::iterator It;
  std::pair<It, It> hitPos;

  if (seed_.id().subdetId() == EcalBarrel) {
    hitPos = std::equal_range(hits_.begin(), hits_.end(), seed_.id(), PairSortByFirst<DetId, float, EBDetIdSorter>());
  } else {
    hitPos = std::equal_range(hits_.begin(), hits_.end(), seed_.id(), PairSortByFirst<DetId, float, EEDetIdSorter>());
  }

  if (hitPos.first == hitPos.second) {  //it doesnt already exist in the vec, add it
    hits_.insert(hitPos.first, std::pair<DetId, float>(seed_.id(), 1.));
    energy_ += seed_.energy();
    containsSeed_ = true;

    return true;
  } else
    return false;
}

bool Multi5x5ClusterAlgo::ProtoBasicCluster::isSeedCrysInHits_() const {
  for (auto const& hit : hits_) {
    if (seed_.id() == hit.first)
      return true;
  }
  return false;
}
