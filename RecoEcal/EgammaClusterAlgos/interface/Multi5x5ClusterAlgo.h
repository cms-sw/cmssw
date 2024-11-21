#ifndef RecoECAL_ECALClusters_Multi5x5ClusterAlgo_h
#define RecoECAL_ECALClusters_Multi5x5ClusterAlgo_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>
#include <optional>

typedef std::map<DetId, EcalRecHit> RecHitsMap;

class Multi5x5ClusterAlgo {
public:
  //the 5x5 clustering algo by default makes basic clusters which may not contain their seed crystal if they are close by to other clusters
  //however we would like to post-fix the basic clusters to ensure they always contain their seed crystal
  //so we define a proto basic cluster class which contains all the information which would be in a basic cluster
  //which allows the addition of its seed and the removal of a seed of another cluster easily
  class ProtoBasicCluster {
    std::vector<std::pair<DetId, float> > hits_;
    EcalRecHit seed_;
    float energy_;
    bool containsSeed_;

  public:
    ProtoBasicCluster();
    ProtoBasicCluster(float iEnergy, const EcalRecHit &iSeed, std::vector<std::pair<DetId, float> > iHits)
        : hits_(std::move(iHits)), seed_(iSeed), energy_(iEnergy), containsSeed_{isSeedCrysInHits_()} {}

    float energy() const { return energy_; }
    const EcalRecHit &seed() const { return seed_; }
    const std::vector<std::pair<DetId, float> > &hits() const { return hits_; }
    bool containsSeed() const { return containsSeed_; }

    bool removeHit(const EcalRecHit &hitToRM);
    bool addSeed();

  private:
    bool isSeedCrysInHits_() const;
  };

  Multi5x5ClusterAlgo() {}

  Multi5x5ClusterAlgo(double ebst,
                      double ecst,
                      const std::vector<int> &v_chstatus,
                      const PositionCalc &posCalc,
                      bool reassignSeedCrysToClusterItSeeds = false)
      : ecalBarrelSeedThreshold_(ebst),
        ecalEndcapSeedThreshold_(ecst),
        v_chstatus_(v_chstatus),
        reassignSeedCrysToClusterItSeeds_(reassignSeedCrysToClusterItSeeds) {
    posCalculator_ = posCalc;
    std::sort(v_chstatus_.begin(), v_chstatus_.end());
  }

  virtual ~Multi5x5ClusterAlgo() {}

  // this is the method that will start the clusterisation
  std::vector<reco::BasicCluster> makeClusters(
      const EcalRecHitCollection *hits,
      const CaloSubdetectorGeometry *geometry,
      const CaloSubdetectorTopology *topology_p,
      const CaloSubdetectorGeometry *geometryES_p,
      reco::CaloID::Detectors detector,
      bool regional = false,
      const std::vector<RectangularEtaPhiRegion> &regions = std::vector<RectangularEtaPhiRegion>());

  /// point in the space
  typedef math::XYZPoint Point;

private:
  //algo to compute position of clusters
  PositionCalc posCalculator_;

  /// The ecal region used
  reco::CaloID::Detectors detector_;

  // Energy required for a seed:
  double ecalBarrelSeedThreshold_;
  double ecalEndcapSeedThreshold_;

  // recHit flag to be excluded from seeding
  std::vector<int> v_chstatus_;

  bool reassignSeedCrysToClusterItSeeds_;  //the seed of the 5x5 crystal is sometimes in another basic cluster, however we may want to put it back into the cluster it seeds

  std::vector<reco::BasicCluster> mainSearch(const EcalRecHitCollection *hits,
                                             const CaloSubdetectorGeometry *geometry_p,
                                             const CaloSubdetectorTopology *topology_p,
                                             const CaloSubdetectorGeometry *geometryES_p,
                                             const std::vector<EcalRecHit> &seeds);

  // Is the crystal at the navigator position a
  // local maxiumum in energy?
  bool checkMaxima(CaloNavigator<DetId> &navigator, const EcalRecHitCollection *hits) const;

  // prepare the 5x5 taking care over which crystals
  // are allowed to seed new clusters and which are not
  // after the preparation is complete
  std::vector<std::pair<DetId, float> > prepareCluster(CaloNavigator<DetId> &navigator,
                                                       const EcalRecHitCollection *hits,
                                                       const CaloSubdetectorGeometry *geometry,
                                                       std::set<DetId> &used_seeds,
                                                       std::set<DetId> &canSeed_s) const;

  // Add the crystal with DetId det to the current
  // vector of crystals if it meets certain criteria
  static bool addCrystal(const DetId &det, const EcalRecHitCollection &recHits);

  // take the crystals in the current_v and build
  // them into a BasicCluster
  // NOTE: this can't be const because of posCalculator_
  std::optional<ProtoBasicCluster> makeCluster(const EcalRecHitCollection *hits,
                                               const CaloSubdetectorGeometry *geometry_p,
                                               const CaloSubdetectorGeometry *geometryES_p,
                                               const EcalRecHitCollection::const_iterator &seedIt,
                                               bool seedOutside,
                                               std::vector<std::pair<DetId, float> > &current_v);
};

#endif
