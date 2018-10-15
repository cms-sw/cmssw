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

typedef std::map<DetId, EcalRecHit> RecHitsMap;

class Multi5x5ClusterAlgo 
{
    public:
  //the 5x5 clustering algo by default makes basic clusters which may not contain their seed crystal if they are close by to other clusters 
  //however we would like to post-fix the basic clusters to ensure they always contain their seed crystal
  //so we define a proto basic cluster class which contains all the information which would be in a basic cluster 
  //which allows the addition of its seed and the removal of a seed of another cluster easily
  class ProtoBasicCluster {
    float energy_;
    EcalRecHit seed_;
    std::vector<std::pair<DetId,float> > hits_;
    bool containsSeed_;
  public:
    ProtoBasicCluster();
    ProtoBasicCluster(float iEnergy,const EcalRecHit& iSeed,std::vector<std::pair<DetId,float> >& iHits):energy_(iEnergy),seed_(iSeed){hits_.swap(iHits);containsSeed_=isSeedCrysInHits_();}
    
    float energy()const{return energy_;}
    const EcalRecHit& seed()const{return seed_;}
    const std::vector<std::pair<DetId,float> >& hits()const{return hits_;}
    bool containsSeed()const{return containsSeed_;}

    bool removeHit(const EcalRecHit& hitToRM);
    bool addSeed();
  private:
    bool isSeedCrysInHits_()const; 

  };
    

  Multi5x5ClusterAlgo() {
        }

  Multi5x5ClusterAlgo(double ebst, double ecst, const std::vector<int>& v_chstatus, const PositionCalc& posCalc,bool reassignSeedCrysToClusterItSeeds=false) : 
	  ecalBarrelSeedThreshold(ebst), ecalEndcapSeedThreshold(ecst),  v_chstatus_(v_chstatus) ,reassignSeedCrysToClusterItSeeds_(reassignSeedCrysToClusterItSeeds) {
                posCalculator_ = posCalc;
                std::sort( v_chstatus_.begin(), v_chstatus_.end() );
            }

        virtual ~Multi5x5ClusterAlgo()
        {
        }


        // this is the method that will start the clusterisation
        std::vector<reco::BasicCluster> makeClusters(const EcalRecHitCollection* hits,
                const CaloSubdetectorGeometry *geometry,
                const CaloSubdetectorTopology *topology_p,
                const CaloSubdetectorGeometry *geometryES_p,
                reco::CaloID::Detectors detector,
                bool regional = false,
                const std::vector<RectangularEtaPhiRegion>& regions = std::vector<RectangularEtaPhiRegion>());

        /// point in the space
        typedef math::XYZPoint Point;

    private: 

        //algo to compute position of clusters
        PositionCalc posCalculator_;

        /// The ecal region used
        reco::CaloID::Detectors detector_;

        // Energy required for a seed:
        double ecalBarrelSeedThreshold;
        double ecalEndcapSeedThreshold;

        // collection of all rechits
        const EcalRecHitCollection *recHits_;

        // The vector of seeds:
        std::vector<EcalRecHit> seeds;

        std::vector<std::pair<DetId,int> > whichClusCrysBelongsTo_;

        // The set of used DetID's
        std::set<DetId> used_s;
        std::set<DetId> canSeed_s; // set of crystals not to be added but which can seed
        // a new 3x3 (e.g. the outer crystals in a 5x5)


        // The vector of DetId's in the cluster currently reconstructed
        std::vector<std::pair<DetId, float> > current_v;

        // The vector of clusters
        std::vector<reco::BasicCluster> clusters_v;
        std::vector<ProtoBasicCluster> protoClusters_; 
        // recHit flag to be excluded from seeding
        std::vector<int> v_chstatus_; 

        bool reassignSeedCrysToClusterItSeeds_; //the seed of the 5x5 crystal is sometimes in another basic cluster, however we may want to put it back into the cluster it seeds

 

        void mainSearch(const EcalRecHitCollection* hits,
                const CaloSubdetectorGeometry *geometry_p,
                const CaloSubdetectorTopology *topology_p,
                const CaloSubdetectorGeometry *geometryES_p);

        // Is the crystal at the navigator position a 
        // local maxiumum in energy?
        bool checkMaxima(CaloNavigator<DetId> &navigator,
                const EcalRecHitCollection *hits);

        // prepare the 5x5 taking care over which crystals
        // are allowed to seed new clusters and which are not
        // after the preparation is complete
        void prepareCluster(CaloNavigator<DetId> &navigator,
                const EcalRecHitCollection *hits,
                const CaloSubdetectorGeometry *geometry);

        // Add the crystal with DetId det to the current
        // vector of crystals if it meets certain criteria
        void addCrystal(const DetId &det);


        // take the crystals in the current_v and build 
        // them into a BasicCluster
        void makeCluster(const EcalRecHitCollection* hits,
                const CaloSubdetectorGeometry *geometry_p,
                const CaloSubdetectorGeometry *geometryES_p, 
                const EcalRecHitCollection::const_iterator &seedIt,
                bool seedOutside);

};

#endif
