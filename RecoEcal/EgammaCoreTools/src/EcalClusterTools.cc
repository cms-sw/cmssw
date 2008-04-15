#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"


std::pair<DetId, float> EcalClusterTools::getMaximum( const std::vector<DetId> &v_id, const EcalRecHitCollection *recHits)
{
        float max = 0;
        DetId id(0);
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                float energy = recHitEnergy( v_id[i], recHits );
                if ( energy > max ) {
                        max = energy;
                        id = v_id[i];
                }
        }
        return std::pair<DetId, float>(id, max);
}



float EcalClusterTools::recHitEnergy(DetId id, const EcalRecHitCollection *recHits)
{
        if ( id == DetId(0) ) {
                return 0;
        } else {
                EcalRecHitCollection::const_iterator it = recHits->find( id );
                if ( it != recHits->end() ) {
                        return (*it).energy();
                } else {
                        // FIXME : throw exception !
                }
        }
        return 0;
}



float EcalClusterTools::matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology );
        float energy = 0;
        for ( int i = ixMin; i <= ixMax; ++i ) {
                for ( int j = iyMin; j <= iyMax; ++j ) {
                        cursor.home();
                        cursor.offsetBy( i, j );
                        energy += recHitEnergy( *cursor, recHits );
                }
        }
        return energy;
}



float EcalClusterTools::e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 0 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 0 ) );
        energies.sort();
        return *--energies.end();
}



float EcalClusterTools::e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 0 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 1 ) );
        energies.sort();
        return *--energies.end();
}



float EcalClusterTools::e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 1 );
}



float EcalClusterTools::e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -2, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -2, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -1, 2 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -1, 2 ) );
        return *--energies.end();
}



float EcalClusterTools::e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, 2 );
}



float EcalClusterTools::eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        return getMaximum( cluster.getHitsByDetId(), recHits ).second;
}



float EcalClusterTools::e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        std::list<float> energies;
        if ( v_id.size() < 2 ) return 0;
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                energies.push_back( recHitEnergy( v_id[i], recHits ) );
        }
        return *--(--energies.end());
}
