#ifndef RecoECAL_ECALClusters_Multi5x5ClusterAlgo_h
#define RecoECAL_ECALClusters_Multi5x5ClusterAlgo_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalRecHitLess.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

typedef std::map<DetId, EcalRecHit> RecHitsMap;

class Multi5x5ClusterAlgo 
{
    public:


        Multi5x5ClusterAlgo() {
        }

        Multi5x5ClusterAlgo(double ebst, double ecst, std::vector<int> v_chstatus, const PositionCalc& posCalc) : 
            ecalBarrelSeedThreshold(ebst), ecalEndcapSeedThreshold(ecst),  v_chstatus_(v_chstatus) {
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
                const std::vector<EcalEtaPhiRegion>& regions = std::vector<EcalEtaPhiRegion>());

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

        // The set of used DetID's
        std::set<DetId> used_s;
        std::set<DetId> canSeed_s; // set of crystals not to be added but which can seed
        // a new 3x3 (e.g. the outer crystals in a 5x5)


        // The vector of DetId's in the cluster currently reconstructed
        std::vector<std::pair<DetId, float> > current_v;

        // The vector of clusters
        std::vector<reco::BasicCluster> clusters_v;

        // recHit flag to be excluded from seeding
        std::vector<int> v_chstatus_;

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
