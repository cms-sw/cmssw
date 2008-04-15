#ifndef RecoEcal_EgammaCoreTools_EcalClusterTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterTools_h

/** \class EcalClusterTools
 *  
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 * 
 * \version $Id: 
 *
 */

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class DetId;
class CaloSubdetectorTopology;

class EcalClusterTools {
        public:
                EcalClusterTools() {};
                ~EcalClusterTools() {};
                float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
                float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
                float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
                float eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
                float e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )

        private:
                std::pair<DetId, float> getMaximum( const std::vector<DetId> &v_id, const EcalRecHitCollection *recHits);
                float recHitEnergy(DetId id, const EcalRecHitCollection *recHits);
                float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
};

#endif
