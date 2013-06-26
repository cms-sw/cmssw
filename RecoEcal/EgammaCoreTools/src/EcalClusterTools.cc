#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "CLHEP/Geometry/Transform3D.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"

float EcalClusterTools::getFraction( const std::vector< std::pair<DetId, float> > &v_id, DetId id
			  ){
  float frac = 0.0;
  for ( size_t i = 0; i < v_id.size(); ++i ) {
    if(v_id[i].first.rawId()==id.rawId()){
      frac=v_id[i].second;
    }
  }
  return frac;
}


std::pair<DetId, float> EcalClusterTools::getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits)
{
    float max = 0;
    DetId id(0);
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        float energy = recHitEnergy( v_id[i].first, recHits ) * v_id[i].second;
        if ( energy > max ) {
            max = energy;
            id = v_id[i].first;
        }
    }
    return std::pair<DetId, float>(id, max);
}

std::pair<DetId, float> EcalClusterTools::getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits,const std::vector<int>& flagsexcl,  const std::vector<int>& severitiesexcl, const  EcalSeverityLevelAlgo *sevLv)
{
    float max = 0;
    DetId id(0);
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        float energy = recHitEnergy( v_id[i].first, recHits,flagsexcl, severitiesexcl, sevLv ) * v_id[i].second;
        if ( energy > max ) {
            max = energy;
            id = v_id[i].first;
        }
    }
    return std::pair<DetId, float>(id, max);
}


std::pair<DetId, float> EcalClusterTools::getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits)
{
    return getMaximum( cluster.hitsAndFractions(), recHits );
}

std::pair<DetId, float> EcalClusterTools::getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits,const std::vector<int>& flagsexcl,  const std::vector<int>& severitiesexcl, const  EcalSeverityLevelAlgo *sevLv )
{
    return getMaximum( cluster.hitsAndFractions(), recHits, flagsexcl, severitiesexcl, sevLv );
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
            //throw cms::Exception("EcalRecHitNotFound") << "The recHit corresponding to the DetId" << id.rawId() << " not found in the EcalRecHitCollection";
            // the recHit is not in the collection (hopefully zero suppressed)
            return 0;
        }
    }
    return 0;
}

float EcalClusterTools::recHitEnergy(DetId id, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const  EcalSeverityLevelAlgo *sevLv)
{
    if ( id == DetId(0) ) {
        return 0;
    } else {
        EcalRecHitCollection::const_iterator it = recHits->find( id );
        if ( it != recHits->end() ) {
	  // avoid anomalous channels (recoFlag based)
	  uint32_t rhFlag = (*it).recoFlag();
	  std::vector<int>::const_iterator vit = std::find( flagsexcl.begin(), flagsexcl.end(), rhFlag );
	  //if your flag was found to be one which is excluded, zero out
	  //this energy.
	  if ( vit != flagsexcl.end() ) return 0;
	    
	  int severityFlag =  sevLv->severityLevel( it->id(), *recHits);
	  std::vector<int>::const_iterator sit = std::find(severitiesexcl.begin(), severitiesexcl.end(), severityFlag);
	  //if you were flagged by some condition (kWeird etc.)
	  //zero out this energy.
	  if (sit!= severitiesexcl.end())
	    return 0; 
	  //If we make it here, you're a found, clean hit.
	  return (*it).energy();
        } else {
	  //throw cms::Exception("EcalRecHitNotFound") << "The recHit corresponding to the DetId" << id.rawId() << " not found in the EcalRecHitCollection";
	  // the recHit is not in the collection (hopefully zero suppressed)
	  return 0;
        }
    }
    return 0;
}


// Returns the energy in a rectangle of crystals
// specified in eta by ixMin and ixMax
//       and in phi by iyMin and iyMax
//
// Reference picture (X=seed crystal)
//    iy ___________
//     2 |_|_|_|_|_|
//     1 |_|_|_|_|_|
//     0 |_|_|X|_|_|
//    -1 |_|_|_|_|_|
//    -2 |_|_|_|_|_|
//      -2 -1 0 1 2 ix
float EcalClusterTools::matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
  //take into account fractions
    // fast version
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    float energy = 0;
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
	  cursor.home();
	  cursor.offsetBy( i, j );
	  float frac=getFraction(v_id,*cursor);
	  energy += recHitEnergy( *cursor, recHits )*frac;
        }
    }
    // slow elegant version
    //float energy = 0;
    //std::vector<DetId> v_id = matrixDetId( topology, id, ixMin, ixMax, iyMin, iyMax );
    //for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
    //        energy += recHitEnergy( *it, recHits );
    //}
    return energy;
}

float EcalClusterTools::matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    // fast version
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    float energy = 0;
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
            cursor.home();
            cursor.offsetBy( i, j );
            energy += recHitEnergy( *cursor, recHits, flagsexcl, severitiesexcl, sevLv );
        }
    }
    return energy;
}



std::vector<DetId> EcalClusterTools::matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    std::vector<DetId> v;
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
            cursor.home();
            cursor.offsetBy( i, j );
            if ( *cursor != DetId(0) ) v.push_back( *cursor );
        }
    }
    return v;
}



float EcalClusterTools::e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 0 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0,  0, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1,  0, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 0 ) );


    return *std::max_element(energies.begin(),energies.end());

}


float EcalClusterTools::e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 0,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0,  0, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1,  0, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 0,flagsexcl, severitiesexcl, sevLv ) );


    return *std::max_element(energies.begin(),energies.end());

}


float EcalClusterTools::e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 0 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1,  0, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 1 ) );
    return *std::max_element(energies.begin(),energies.end());
}

float EcalClusterTools::e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 0,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1,  0, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 1,flagsexcl, severitiesexcl, sevLv ) );
    return *std::max_element(energies.begin(),energies.end());
}

float EcalClusterTools::e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 1 );
}


float EcalClusterTools::e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 1,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -2, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -2, 1 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -1, 2 ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -1, 2 ) );
    return *std::max_element(energies.begin(),energies.end());
}

float EcalClusterTools::e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -2, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -2, 1,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -1, 2,flagsexcl, severitiesexcl, sevLv ) );
    energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -1, 2,flagsexcl, severitiesexcl, sevLv ) );
    return *std::max_element(energies.begin(),energies.end());
}



float EcalClusterTools::e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, 2 );
}

float EcalClusterTools::e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, 2,flagsexcl, severitiesexcl, sevLv );
}

float EcalClusterTools::eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    return getMaximum( cluster.hitsAndFractions(), recHits ).second;
}

float EcalClusterTools::eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv  )
{
    return getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).second;
}


float EcalClusterTools::e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::list<float> energies;
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id.size() < 2 ) return 0;
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        energies.push_back( recHitEnergy( v_id[i].first, recHits ) * v_id[i].second );
    }
    energies.sort(); 	         
    return *--(--energies.end());


}

float EcalClusterTools::e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    std::list<float> energies;
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id.size() < 2 ) return 0;
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        energies.push_back( recHitEnergy( v_id[i].first, recHits,flagsexcl, severitiesexcl, sevLv ) * v_id[i].second );
    }
    energies.sort(); 	         
    return *--(--energies.end());


}



float EcalClusterTools::e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 2, -2, 2 );
}


float EcalClusterTools::e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 2, -2, 2,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, -1, -2, 2 );
}

float EcalClusterTools::e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, -1, -2, 2,flagsexcl, severitiesexcl, sevLv );
}


// 
float EcalClusterTools::e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 1, 2 );
}

float EcalClusterTools::e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 1, 2,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, -1 );
}

float EcalClusterTools::e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv  ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, -1,flagsexcl, severitiesexcl, sevLv );
}

// Energy in 2x5 strip containing the max crystal.
// Adapted from code by Sam Harper
float EcalClusterTools::e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id =      getMaximum( cluster.hitsAndFractions(), recHits ).first;

    // 1x5 strip left of seed
    float left   = matrixEnergy( cluster, recHits, topology, id, -1, -1, -2, 2 );
    // 1x5 strip right of seed
    float right  = matrixEnergy( cluster, recHits, topology, id,  1,  1, -2, 2 );
    // 1x5 strip containing seed
    float centre = matrixEnergy( cluster, recHits, topology, id,  0,  0, -2, 2 );

    // Return the maximum of (left+center) or (right+center) strip
    return left > right ? left+centre : right+centre;
}

float EcalClusterTools::e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id =      getMaximum( cluster.hitsAndFractions(), recHits,flagsexcl, severitiesexcl, sevLv ).first;

    // 1x5 strip left of seed
    float left   = matrixEnergy( cluster, recHits, topology, id, -1, -1, -2, 2,flagsexcl, severitiesexcl, sevLv );
    // 1x5 strip right of seed
    float right  = matrixEnergy( cluster, recHits, topology, id,  1,  1, -2, 2,flagsexcl, severitiesexcl, sevLv );
    // 1x5 strip containing seed
    float centre = matrixEnergy( cluster, recHits, topology, id,  0,  0, -2, 2,flagsexcl, severitiesexcl, sevLv );

    // Return the maximum of (left+center) or (right+center) strip
    return left > right ? left+centre : right+centre;
}


float EcalClusterTools::e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -2, 2 );
}

float EcalClusterTools::e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv)
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -2, 2 ,
			 flagsexcl, severitiesexcl, sevLv);
}



float EcalClusterTools::e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 0, 0 );
}

float EcalClusterTools::e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 0, 0,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, 1 );
}

float EcalClusterTools::e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, 1,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, 0, 0 );
}

float EcalClusterTools::e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, 0, 0,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, -1, 0, 0 );
}

float EcalClusterTools::eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, -1, 0, 0,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 1, 0, 0 );
}

float EcalClusterTools::eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 1, 0, 0,flagsexcl, severitiesexcl, sevLv );
}


float EcalClusterTools::eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, 1, 1 );
}

float EcalClusterTools::eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, 1, 1,flagsexcl, severitiesexcl, sevLv );
}



float EcalClusterTools::eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, -1 );
}

float EcalClusterTools::eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, -1,flagsexcl, severitiesexcl, sevLv  );
}

std::vector<float> EcalClusterTools::energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::vector<float> basketFraction( 2 * EBDetId::kModulesPerSM );
    float clusterEnergy = cluster.energy();
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterTools::energyBasketFractionEta") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        basketFraction[ EBDetId(v_id[i].first).im()-1 + EBDetId(v_id[i].first).positiveZ()*EBDetId::kModulesPerSM ] += recHitEnergy( v_id[i].first, recHits ) * v_id[i].second / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}


std::vector<float> EcalClusterTools::energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    std::vector<float> basketFraction( 2 * EBDetId::kModulesPerSM );
    float clusterEnergy = cluster.energy();
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterTools::energyBasketFractionEta") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        basketFraction[ EBDetId(v_id[i].first).im()-1 + EBDetId(v_id[i].first).positiveZ()*EBDetId::kModulesPerSM ] += recHitEnergy( v_id[i].first, recHits,flagsexcl, severitiesexcl, sevLv ) * v_id[i].second / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}

std::vector<float> EcalClusterTools::energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::vector<float> basketFraction( 2 * (EBDetId::MAX_IPHI / EBDetId::kCrystalsInPhi) );
    float clusterEnergy = cluster.energy();
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterTools::energyBasketFractionPhi") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        basketFraction[ (EBDetId(v_id[i].first).iphi()-1)/EBDetId::kCrystalsInPhi + EBDetId(v_id[i].first).positiveZ()*EBDetId::kTowersInPhi] += recHitEnergy( v_id[i].first, recHits ) * v_id[i].second / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}


std::vector<float> EcalClusterTools::energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    std::vector<float> basketFraction( 2 * (EBDetId::MAX_IPHI / EBDetId::kCrystalsInPhi) );
    float clusterEnergy = cluster.energy();
    std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterTools::energyBasketFractionPhi") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        basketFraction[ (EBDetId(v_id[i].first).iphi()-1)/EBDetId::kCrystalsInPhi + EBDetId(v_id[i].first).positiveZ()*EBDetId::kTowersInPhi] += recHitEnergy( v_id[i].first, recHits,flagsexcl, severitiesexcl, sevLv ) * v_id[i].second / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}


std::vector<EcalClusterTools::EcalClusterEnergyDeposition> EcalClusterTools::getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 )
{
    std::vector<EcalClusterTools::EcalClusterEnergyDeposition> energyDistribution;
    // init a map of the energy deposition centered on the
    // cluster centroid. This is for momenta calculation only.
    CLHEP::Hep3Vector clVect(cluster.position().x(), cluster.position().y(), cluster.position().z());
    CLHEP::Hep3Vector clDir(clVect);
    clDir*=1.0/clDir.mag();
    // in the transverse plane, axis perpendicular to clusterDir
    CLHEP::Hep3Vector theta_axis(clDir.y(),-clDir.x(),0.0);
    theta_axis *= 1.0/theta_axis.mag();
    CLHEP::Hep3Vector phi_axis = theta_axis.cross(clDir);

    std::vector< std::pair<DetId, float> > clusterDetIds = cluster.hitsAndFractions();

    EcalClusterEnergyDeposition clEdep;
    EcalRecHit testEcalRecHit;
    std::vector< std::pair<DetId, float> >::iterator posCurrent;
    // loop over crystals
    for(posCurrent=clusterDetIds.begin(); posCurrent!=clusterDetIds.end(); ++posCurrent) {
        EcalRecHitCollection::const_iterator itt = recHits->find( (*posCurrent).first );
        testEcalRecHit=*itt;

        if(( (*posCurrent).first != DetId(0)) && (recHits->find( (*posCurrent).first ) != recHits->end())) {
            clEdep.deposited_energy = testEcalRecHit.energy() * (*posCurrent).second;
            // if logarithmic weight is requested, apply cut on minimum energy of the recHit
            if(logW) {
                //double w0 = parameterMap_.find("W0")->second;

                double weight = std::max(0.0, w0 + log(fabs(clEdep.deposited_energy)/cluster.energy()) );
                if(weight==0) {
                    LogDebug("ClusterShapeAlgo") << "Crystal has insufficient energy: E = " 
                        << clEdep.deposited_energy << " GeV; skipping... ";
                    continue;
                }
                else LogDebug("ClusterShapeAlgo") << "===> got crystal. Energy = " << clEdep.deposited_energy << " GeV. ";
            }
            DetId id_ = (*posCurrent).first;
            const CaloCellGeometry *this_cell = geometry->getSubdetectorGeometry(id_)->getGeometry(id_);
            GlobalPoint cellPos = this_cell->getPosition();
            CLHEP::Hep3Vector gblPos (cellPos.x(),cellPos.y(),cellPos.z()); //surface position?
            // Evaluate the distance from the cluster centroid
            CLHEP::Hep3Vector diff = gblPos - clVect;
            // Important: for the moment calculation, only the "lateral distance" is important
            // "lateral distance" r_i = distance of the digi position from the axis Origin-Cluster Center
            // ---> subtract the projection on clDir
            CLHEP::Hep3Vector DigiVect = diff - diff.dot(clDir)*clDir;
            clEdep.r = DigiVect.mag();
            LogDebug("ClusterShapeAlgo") << "E = " << clEdep.deposited_energy
                << "\tdiff = " << diff.mag()
                << "\tr = " << clEdep.r;
            clEdep.phi = DigiVect.angle(theta_axis);
            if(DigiVect.dot(phi_axis)<0) clEdep.phi = 2 * M_PI - clEdep.phi;
            energyDistribution.push_back(clEdep);
        }
    } 
    return energyDistribution;
}



std::vector<float> EcalClusterTools::lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 )
{
    std::vector<EcalClusterTools::EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );

    std::vector<float> lat;
    double r, redmoment=0;
    double phiRedmoment = 0 ;
    double etaRedmoment = 0 ;
    int n,n1,n2,tmp;
    int clusterSize=energyDistribution.size();
    float etaLat_, phiLat_, lat_;
    if (clusterSize<3) {
        etaLat_ = 0.0 ; 
        lat_ = 0.0;
        lat.push_back(0.);
        lat.push_back(0.);
        lat.push_back(0.);
        return lat; 
    }

    n1=0; n2=1;
    if (energyDistribution[1].deposited_energy > 
            energyDistribution[0].deposited_energy) 
    {
        tmp=n2; n2=n1; n1=tmp;
    }
    for (int i=2; i<clusterSize; i++) {
        n=i;
        if (energyDistribution[i].deposited_energy > 
                energyDistribution[n1].deposited_energy) 
        {
            tmp = n2;
            n2 = n1; n1 = i; n=tmp;
        } else {
            if (energyDistribution[i].deposited_energy > 
                    energyDistribution[n2].deposited_energy) 
            {
                tmp=n2; n2=i; n=tmp;
            }
        }

        r = energyDistribution[n].r;
        redmoment += r*r* energyDistribution[n].deposited_energy;
        double rphi = r * cos (energyDistribution[n].phi) ;
        phiRedmoment += rphi * rphi * energyDistribution[n].deposited_energy;
        double reta = r * sin (energyDistribution[n].phi) ;
        etaRedmoment += reta * reta * energyDistribution[n].deposited_energy;
    } 
    double e1 = energyDistribution[n1].deposited_energy;
    double e2 = energyDistribution[n2].deposited_energy;

    lat_ = redmoment/(redmoment+2.19*2.19*(e1+e2));
    phiLat_ = phiRedmoment/(phiRedmoment+2.19*2.19*(e1+e2));
    etaLat_ = etaRedmoment/(etaRedmoment+2.19*2.19*(e1+e2));

    lat.push_back(etaLat_);
    lat.push_back(phiLat_);
    lat.push_back(lat_);
    return lat;
}



math::XYZVector EcalClusterTools::meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry )
{
    // find mean energy position of a 5x5 cluster around the maximum
    math::XYZVector meanPosition(0.0, 0.0, 0.0);
    std::vector<DetId> v_id = matrixDetId( topology, getMaximum( cluster, recHits ).first, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
        GlobalPoint positionGP = geometry->getSubdetectorGeometry( *it )->getGeometry( *it )->getPosition();
        math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
        meanPosition = meanPosition + recHitEnergy( *it, recHits ) * position;
    }
    return meanPosition / e5x5( cluster, recHits, topology );
}

//================================================= meanClusterPosition===================================================================================

math::XYZVector EcalClusterTools::meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv )
{
    // find mean energy position of a 5x5 cluster around the maximum
    math::XYZVector meanPosition(0.0, 0.0, 0.0);
    std::vector<DetId> v_id = matrixDetId( topology, getMaximum( cluster, recHits ).first, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
        GlobalPoint positionGP = geometry->getSubdetectorGeometry( *it )->getGeometry( *it )->getPosition();
        math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
        meanPosition = meanPosition + recHitEnergy( *it, recHits,flagsexcl, severitiesexcl, sevLv ) * position;
    }
    return meanPosition / e5x5( cluster, recHits, topology,flagsexcl, severitiesexcl, sevLv );
}


//returns mean energy weighted eta/phi in crystals from the seed
//iPhi is not defined for endcap and is returned as zero
//return <eta,phi>
//we have an issue in working out what to do for negative energies
//I (Sam Harper) think it makes sense to ignore crystals with E<0 in the calculation as they are ignored
//in the sigmaIEtaIEta calculation (well they arent fully ignored, they do still contribute to the e5x5 sum
//in the sigmaIEtaIEta calculation but not here)
std::pair<float,float>  EcalClusterTools::mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;
    float meanDEta=0.;
    float meanDPhi=0.;
    float energySum=0.;

    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
        float energy = recHitEnergy(*it,recHits);
        if(energy<0.) continue;//skipping negative energy crystals
        meanDEta += energy * getNrCrysDiffInEta(*it,seedId);
        meanDPhi += energy * getNrCrysDiffInPhi(*it,seedId);	
        energySum +=energy;
    }
    meanDEta /=energySum;
    meanDPhi /=energySum;
    return std::pair<float,float>(meanDEta,meanDPhi);
}

std::pair<float,float>  EcalClusterTools::mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;
    float meanDEta=0.;
    float meanDPhi=0.;
    float energySum=0.;

    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
        float energy = recHitEnergy(*it,recHits,flagsexcl, severitiesexcl, sevLv);
        if(energy<0.) continue;//skipping negative energy crystals
        meanDEta += energy * getNrCrysDiffInEta(*it,seedId);
        meanDPhi += energy * getNrCrysDiffInPhi(*it,seedId);	
        energySum +=energy;
    }
    meanDEta /=energySum;
    meanDPhi /=energySum;
    return std::pair<float,float>(meanDEta,meanDPhi);
}

//returns mean energy weighted x/y in normalised crystal coordinates
//only valid for endcap, returns 0,0 for barrel
//we have an issue in working out what to do for negative energies
//I (Sam Harper) think it makes sense to ignore crystals with E<0 in the calculation as they are ignored
//in the sigmaIEtaIEta calculation (well they arent fully ignored, they do still contribute to the e5x5 sum
//in the sigmaIEtaIEta calculation but not here)
std::pair<float,float> EcalClusterTools::mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;

    std::pair<float,float> meanXY(0.,0.);
    if(seedId.subdetId()==EcalBarrel) return meanXY;

    float energySum=0.;

    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
        float energy = recHitEnergy(*it,recHits);
        if(energy<0.) continue;//skipping negative energy crystals
        meanXY.first += energy * getNormedIX(*it);
        meanXY.second += energy * getNormedIY(*it);
        energySum +=energy;
    }
    meanXY.first/=energySum;
    meanXY.second/=energySum;
    return meanXY;
}

std::pair<float,float> EcalClusterTools::mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;

    std::pair<float,float> meanXY(0.,0.);
    if(seedId.subdetId()==EcalBarrel) return meanXY;

    float energySum=0.;

    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
        float energy = recHitEnergy(*it,recHits,flagsexcl, severitiesexcl, sevLv);
        if(energy<0.) continue;//skipping negative energy crystals
        meanXY.first += energy * getNormedIX(*it);
        meanXY.second += energy * getNormedIY(*it);
        energySum +=energy;
    }
    meanXY.first/=energySum;
    meanXY.second/=energySum;
    return meanXY;
}


std::vector<float> EcalClusterTools::covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry, float w0)
{
    float e_5x5 = e5x5( cluster, recHits, topology );
    float covEtaEta, covEtaPhi, covPhiPhi;
    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        math::XYZVector meanPosition = meanClusterPosition( cluster, recHits, topology, geometry );

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        DetId id = getMaximum( v_id, recHits ).first;
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
        for ( int i = -2; i <= 2; ++i ) {
            for ( int j = -2; j <= 2; ++j ) {
                cursor.home();
                cursor.offsetBy( i, j );
                float energy = recHitEnergy( *cursor, recHits );

                if ( energy <= 0 ) continue;

                GlobalPoint position = geometry->getSubdetectorGeometry(*cursor)->getGeometry(*cursor)->getPosition();

                double dPhi = position.phi() - meanPosition.phi();
                if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
                if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }

                double dEta = position.eta() - meanPosition.eta();
                double w = 0.;
                w = std::max(0.0, w0 + log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            }
        }

        if (denominator != 0.0) {
            covEtaEta =  numeratorEtaEta / denominator;
            covEtaPhi =  numeratorEtaPhi / denominator;
            covPhiPhi =  numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}

//==================================================== Covariances===========================================================================

std::vector<float> EcalClusterTools::covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry,const std::vector<int>& flagsexcl,const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv, float w0)
{
    float e_5x5 = e5x5( cluster, recHits, topology,flagsexcl, severitiesexcl, sevLv );
    float covEtaEta, covEtaPhi, covPhiPhi;
    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        math::XYZVector meanPosition = meanClusterPosition( cluster, recHits, topology, geometry,flagsexcl, severitiesexcl, sevLv );

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        DetId id = getMaximum( v_id, recHits ).first;
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
        for ( int i = -2; i <= 2; ++i ) {
            for ( int j = -2; j <= 2; ++j ) {
                cursor.home();
                cursor.offsetBy( i, j );
                float energy = recHitEnergy( *cursor, recHits,flagsexcl, severitiesexcl, sevLv );

                if ( energy <= 0 ) continue;

                GlobalPoint position = geometry->getSubdetectorGeometry(*cursor)->getGeometry(*cursor)->getPosition();

                double dPhi = position.phi() - meanPosition.phi();
                if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
                if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }

                double dEta = position.eta() - meanPosition.eta();
                double w = 0.;
                w = std::max(0.0, w0 + log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            }
        }

        if (denominator != 0.0) {
            covEtaEta =  numeratorEtaEta / denominator;
            covEtaPhi =  numeratorEtaPhi / denominator;
            covPhiPhi =  numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}





//for covIEtaIEta,covIEtaIPhi and covIPhiIPhi are defined but only covIEtaIEta has been actively studied
//instead of using absolute eta/phi it counts crystals normalised so that it gives identical results to normal covariances except near the cracks where of course its better 
//it also does not require any eta correction function in the endcap
//it is multipled by an approprate crystal size to ensure it gives similar values to covariances(...)
std::vector<float> EcalClusterTools::localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology,float w0)
{

    float e_5x5 = e5x5( cluster, recHits, topology );
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord( cluster, recHits, topology );
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology);

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        //these allow us to scale the localCov by the crystal size 
        //so that the localCovs have the same average value as the normal covs
        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum( v_id, recHits ).first;

        bool isBarrel=seedId.subdetId()==EcalBarrel;
        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( seedId, topology->getSubdetectorTopology( seedId ) );

        for ( int eastNr = -2; eastNr <= 2; ++eastNr ) { //east is eta in barrel
            for ( int northNr = -2; northNr <= 2; ++northNr ) { //north is phi in barrel
                cursor.home();
                cursor.offsetBy( eastNr, northNr);
                float energy = recHitEnergy( *cursor, recHits );
                if ( energy <= 0 ) continue;

                float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
                float dPhi = 0;

                if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
                else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);


                double w = std::max(0.0, w0 + log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            } //end east loop
        }//end north loop


        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }


    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}

//==================================================================localCovariances======================================================================

std::vector<float> EcalClusterTools::localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology,const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv,float w0)
{

    float e_5x5 = e5x5( cluster, recHits, topology,flagsexcl, severitiesexcl, sevLv );
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        std::vector< std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord( cluster, recHits, topology,flagsexcl, severitiesexcl, sevLv );
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology,flagsexcl, severitiesexcl, sevLv);

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        //these allow us to scale the localCov by the crystal size 
        //so that the localCovs have the same average value as the normal covs
        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum( v_id, recHits ).first;

        bool isBarrel=seedId.subdetId()==EcalBarrel;
        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( seedId, topology->getSubdetectorTopology( seedId ) );

        for ( int eastNr = -2; eastNr <= 2; ++eastNr ) { //east is eta in barrel
            for ( int northNr = -2; northNr <= 2; ++northNr ) { //north is phi in barrel
                cursor.home();
                cursor.offsetBy( eastNr, northNr);
                float energy = recHitEnergy( *cursor, recHits,flagsexcl, severitiesexcl, sevLv);
                if ( energy <= 0 ) continue;

                float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
                float dPhi = 0;

                if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
                else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);


                double w = std::max(0.0, w0 + log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            } //end east loop
        }//end north loop


        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }


    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}


double EcalClusterTools::zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0, bool logW, float w0 )
{
    return absZernikeMoment( cluster, recHits, geometry, 2, 0, R0, logW, w0 );
}



double EcalClusterTools::zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0, bool logW, float w0 )
{
    return absZernikeMoment( cluster, recHits, geometry, 4, 2, R0, logW, w0 );
}



double EcalClusterTools::absZernikeMoment( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    // 1. Check if n,m are correctly
    if ((m>n) || ((n-m)%2 != 0) || (n<0) || (m<0)) return -1;

    // 2. Check if n,R0 are within validity Range :
    // n>20 or R0<2.19cm  just makes no sense !
    if ((n>20) || (R0<=2.19)) return -1;
    if (n<=5) return fast_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
    else return calc_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
}


double EcalClusterTools::fast_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    double r,ph,e,Re=0,Im=0;
    double TotalEnergy = cluster.energy();
    int index = (n/2)*(n/2)+(n/2)+m;
    std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
    int clusterSize = energyDistribution.size();
    if(clusterSize < 3) return 0.0;

    for (int i=0; i<clusterSize; i++)
    { 
        r = energyDistribution[i].r / R0;
        if (r<1) {
            std::vector<double> pol;
            pol.push_back( f00(r) );
            pol.push_back( f11(r) );
            pol.push_back( f20(r) );
            pol.push_back( f22(r) );
            pol.push_back( f31(r) );
            pol.push_back( f33(r) );
            pol.push_back( f40(r) );
            pol.push_back( f42(r) );
            pol.push_back( f44(r) );
            pol.push_back( f51(r) );
            pol.push_back( f53(r) );
            pol.push_back( f55(r) );
            ph = (energyDistribution[i]).phi;
            e = energyDistribution[i].deposited_energy;
            Re = Re + e/TotalEnergy * pol[index] * cos( (double) m * ph);
            Im = Im - e/TotalEnergy * pol[index] * sin( (double) m * ph);
        }
    }
    return sqrt(Re*Re+Im*Im);
}



double EcalClusterTools::calc_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    double r, ph, e, Re=0, Im=0, f_nm;
    double TotalEnergy = cluster.energy();
    std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
    int clusterSize=energyDistribution.size();
    if(clusterSize<3) return 0.0;

    for (int i = 0; i < clusterSize; ++i)
    { 
        r = energyDistribution[i].r / R0;
        if (r < 1) {
            ph = energyDistribution[i].phi;
            e = energyDistribution[i].deposited_energy;
            f_nm = 0;
            for (int s=0; s<=(n-m)/2; s++) {
                if (s%2==0) { 
                    f_nm = f_nm + factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                } else {
                    f_nm = f_nm - factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                }
            }
            Re = Re + e/TotalEnergy * f_nm * cos( (double) m*ph);
            Im = Im - e/TotalEnergy * f_nm * sin( (double) m*ph);
        }
    }
    return sqrt(Re*Re+Im*Im);
}

//returns the crystal 'eta' from the det id
//it is defined as the number of crystals from the centre in the eta direction
//for the barrel with its eta/phi geometry it is always integer
//for the endcap it is fractional due to the x/y geometry
float  EcalClusterTools::getIEta(const DetId& id)
{
    if(id.det()==DetId::Ecal){
        if(id.subdetId()==EcalBarrel){
            EBDetId ebId(id);
            return ebId.ieta();
        }else if(id.subdetId()==EcalEndcap){
            float iXNorm = getNormedIX(id);
            float iYNorm = getNormedIY(id);

            return std::sqrt(iXNorm*iXNorm+iYNorm*iYNorm);
        }
    }
    return 0.;    
}


//returns the crystal 'phi' from the det id
//it is defined as the number of crystals from the centre in the phi direction
//for the barrel with its eta/phi geometry it is always integer
//for the endcap it is not defined 
float  EcalClusterTools::getIPhi(const DetId& id)
{
    if(id.det()==DetId::Ecal){
        if(id.subdetId()==EcalBarrel){
            EBDetId ebId(id);
            return ebId.iphi();
        }
    }
    return 0.;    
}

//want to map 1=-50,50=-1,51=1 and 100 to 50 so sub off one if zero or neg
float EcalClusterTools::getNormedIX(const DetId& id)
{
    if(id.det()==DetId::Ecal && id.subdetId()==EcalEndcap){
        EEDetId eeId(id);      
        int iXNorm  = eeId.ix()-50;
        if(iXNorm<=0) iXNorm--;
        return iXNorm;
    }
    return 0;
}

//want to map 1=-50,50=-1,51=1 and 100 to 50 so sub off one if zero or neg
float EcalClusterTools::getNormedIY(const DetId& id)
{
    if(id.det()==DetId::Ecal && id.subdetId()==EcalEndcap){
        EEDetId eeId(id);      
        int iYNorm  = eeId.iy()-50;
        if(iYNorm<=0) iYNorm--;
        return iYNorm;
    }
    return 0;
}

//nr crystals crysId is away from orgin id in eta
float EcalClusterTools::getNrCrysDiffInEta(const DetId& crysId,const DetId& orginId)
{
    float crysIEta = getIEta(crysId);
    float orginIEta = getIEta(orginId);
    bool isBarrel = orginId.subdetId()==EcalBarrel;

    float nrCrysDiff = crysIEta-orginIEta;

    //no iEta=0 in barrel, so if go from positive to negative
    //need to reduce abs(detEta) by 1
    if(isBarrel){ 
        if(crysIEta*orginIEta<0){ // -1 to 1 transition
            if(crysIEta>0) nrCrysDiff--;
            else nrCrysDiff++;
        }
    }
    return nrCrysDiff;
}

//nr crystals crysId is away from orgin id in phi
float EcalClusterTools::getNrCrysDiffInPhi(const DetId& crysId,const DetId& orginId)
{
    float crysIPhi = getIPhi(crysId);
    float orginIPhi = getIPhi(orginId);
    bool isBarrel = orginId.subdetId()==EcalBarrel;

    float nrCrysDiff = crysIPhi-orginIPhi;

    if(isBarrel){ //if barrel, need to map into 0-180 
        if (nrCrysDiff > + 180) { nrCrysDiff = nrCrysDiff - 360; }
        if (nrCrysDiff < - 180) { nrCrysDiff = nrCrysDiff + 360; }
    }
    return nrCrysDiff;
}

//nr crystals crysId is away from mean phi in 5x5 in phi
float EcalClusterTools::getDPhiEndcap(const DetId& crysId,float meanX,float meanY)
{
    float iXNorm  = getNormedIX(crysId);
    float iYNorm  = getNormedIY(crysId);

    float hitLocalR2 = (iXNorm-meanX)*(iXNorm-meanX)+(iYNorm-meanY)*(iYNorm-meanY);
    float hitR2 = iXNorm*iXNorm+iYNorm*iYNorm;
    float meanR2 = meanX*meanX+meanY*meanY;
    float hitR = sqrt(hitR2);
    float meanR = sqrt(meanR2);

    float tmp = (hitR2+meanR2-hitLocalR2)/(2*hitR*meanR);
    if (tmp<-1) tmp =-1;
    if (tmp>1)  tmp=1;
    float phi = acos(tmp);
    float dPhi = hitR*phi;

    return dPhi;
}

std::vector<float> EcalClusterTools::scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, float w0)
{
    const reco::BasicCluster bcluster = *(cluster.seed());

    float e_5x5 = e5x5(bcluster, recHits, topology);
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        std::vector<std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord(bcluster, recHits, topology);
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology);
        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum(v_id, recHits).first;  
        bool isBarrel=seedId.subdetId()==EcalBarrel;

        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        for (size_t i = 0; i < v_id.size(); ++i) {
            CaloNavigator<DetId> cursor = CaloNavigator<DetId>(v_id[i].first, topology->getSubdetectorTopology(v_id[i].first));
            float energy = recHitEnergy(*cursor, recHits);

            if (energy <= 0) continue;

            float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
            float dPhi = 0;
            if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
            else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);



            double w = 0.;
            w = std::max(0.0, w0 + log( energy / e_5x5 ));

            denominator += w;
            numeratorEtaEta += w * dEta * dEta;
            numeratorEtaPhi += w * dEta * dPhi;
            numeratorPhiPhi += w * dPhi * dPhi;
        }

        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        // std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }

    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );

    return v;
}


//================================================================== scLocalCovariances==============================================================

std::vector<float> EcalClusterTools::scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology,const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv, float w0)
{
    const reco::BasicCluster bcluster = *(cluster.seed());

    float e_5x5 = e5x5(bcluster, recHits, topology);
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        std::vector<std::pair<DetId, float> > v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord(bcluster, recHits, topology,flagsexcl, severitiesexcl, sevLv);
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology,flagsexcl, severitiesexcl, sevLv);
        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum(v_id, recHits).first;  
        bool isBarrel=seedId.subdetId()==EcalBarrel;

        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        for (size_t i = 0; i < v_id.size(); ++i) {
            CaloNavigator<DetId> cursor = CaloNavigator<DetId>(v_id[i].first, topology->getSubdetectorTopology(v_id[i].first));
            float energy = recHitEnergy(*cursor, recHits,flagsexcl, severitiesexcl, sevLv);

            if (energy <= 0) continue;

            float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
            float dPhi = 0;
            if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
            else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);



            double w = 0.;
            w = std::max(0.0, w0 + log( energy / e_5x5 ));

            denominator += w;
            numeratorEtaEta += w * dEta * dEta;
            numeratorEtaPhi += w * dEta * dPhi;
            numeratorPhiPhi += w * dPhi * dPhi;
        }

        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        // std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }

    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );

    return v;
}

// compute cluster second moments with respect to principal axes (eigenvectors of sEtaEta, sPhiPhi, sEtaPhi matrix)
// store also angle alpha between major axis and phi.
// takes into account shower elongation in phi direction due to magnetic field effect: 
// default value of 0.8 ensures sMaj = sMin for unconverted photons 
// (if phiCorrectionFactor=1 sMaj > sMin and alpha=0 also for unconverted photons)


Cluster2ndMoments EcalClusterTools::cluster2ndMoments( const reco::BasicCluster &basicCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor, double w0, bool useLogWeights) {

  Cluster2ndMoments returnMoments;
  returnMoments.sMaj = -1.;
  returnMoments.sMin = -1.;
  returnMoments.alpha = 0.;

  // for now implemented only for EB:
  //  if( fabs( basicCluster.eta() ) < 1.479 ) { 

    std::vector<const EcalRecHit*> RH_ptrs;
    
    std::vector< std::pair<DetId, float> > myHitsPair = basicCluster.hitsAndFractions();
    std::vector<DetId> usedCrystals;
    for(unsigned int i=0; i< myHitsPair.size(); i++){
      usedCrystals.push_back(myHitsPair[i].first);
    }
    
    for(unsigned int i=0; i<usedCrystals.size(); i++){
      //get pointer to recHit object
      EcalRecHitCollection::const_iterator myRH = recHits.find(usedCrystals[i]);
      RH_ptrs.push_back(  &(*myRH)  );
    }

      returnMoments = EcalClusterTools::cluster2ndMoments(RH_ptrs, phiCorrectionFactor, w0, useLogWeights);

      //  }

  return returnMoments;

}


Cluster2ndMoments EcalClusterTools::cluster2ndMoments( const reco::SuperCluster &superCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor, double w0, bool useLogWeights) {

  // for now returns second moments of supercluster seed cluster:
  Cluster2ndMoments returnMoments;
  returnMoments.sMaj = -1.;
  returnMoments.sMin = -1.;
  returnMoments.alpha = 0.;

  // for now implemented only for EB:
  //  if( fabs( superCluster.eta() ) < 1.479 ) { 
    returnMoments = EcalClusterTools::cluster2ndMoments( *(superCluster.seed()), recHits, phiCorrectionFactor, w0, useLogWeights);
    //  }

  return returnMoments;

}


Cluster2ndMoments EcalClusterTools::cluster2ndMoments( const std::vector<const EcalRecHit*>& RH_ptrs, double phiCorrectionFactor, double w0, bool useLogWeights) {

  double mid_eta(0),mid_phi(0),mid_x(0),mid_y(0);
  
  double Etot = EcalClusterTools::getSumEnergy(  RH_ptrs  );
 
  double max_phi=-10.;
  double min_phi=100.;
  
  
  std::vector<double> etaDetId;
  std::vector<double> phiDetId;
  std::vector<double> xDetId;
  std::vector<double> yDetId;
  std::vector<double> wiDetId;
 
  unsigned int nCry=0;
  double denominator=0.;
  bool isBarrel(1);

  // loop over rechits and compute weights:
  for(std::vector<const EcalRecHit*>::const_iterator rh_ptr = RH_ptrs.begin(); rh_ptr != RH_ptrs.end(); rh_ptr++){

    //get iEta, iPhi
    double temp_eta(0),temp_phi(0),temp_x(0),temp_y(0);
    isBarrel = (*rh_ptr)->detid().subdetId()==EcalBarrel;
    
    if(isBarrel) {
      temp_eta = (getIEta((*rh_ptr)->detid()) > 0. ? getIEta((*rh_ptr)->detid()) + 84.5 : getIEta((*rh_ptr)->detid()) + 85.5);
      temp_phi= getIPhi((*rh_ptr)->detid()) - 0.5;
    }
    else {
      temp_eta = getIEta((*rh_ptr)->detid());  
      temp_x =  getNormedIX((*rh_ptr)->detid());
      temp_y =  getNormedIY((*rh_ptr)->detid());
    }	  

    double temp_ene=(*rh_ptr)->energy();
    
    double temp_wi=((useLogWeights) ?
                    std::max(0., w0 + log( fabs(temp_ene)/Etot ))
                    :  temp_ene);


    if(temp_phi>max_phi) max_phi=temp_phi;
    if(temp_phi<min_phi) min_phi=temp_phi;
    etaDetId.push_back(temp_eta);
    phiDetId.push_back(temp_phi);
    xDetId.push_back(temp_x);
    yDetId.push_back(temp_y);
    wiDetId.push_back(temp_wi);
    denominator+=temp_wi;
    nCry++;
  }

  if(isBarrel){
    // correct phi wrap-around:
    if(max_phi==359.5 && min_phi==0.5){ 
      for(unsigned int i=0; i<nCry; i++){
	if(phiDetId[i] - 179. > 0.) phiDetId[i]-=360.; 
	mid_phi+=phiDetId[i]*wiDetId[i];
	mid_eta+=etaDetId[i]*wiDetId[i];
      }
    } else{
      for(unsigned int i=0; i<nCry; i++){
	mid_phi+=phiDetId[i]*wiDetId[i];
	mid_eta+=etaDetId[i]*wiDetId[i];
      }
    }
  }else{
    for(unsigned int i=0; i<nCry; i++){
      mid_eta+=etaDetId[i]*wiDetId[i];      
      mid_x+=xDetId[i]*wiDetId[i];
      mid_y+=yDetId[i]*wiDetId[i];
    }
  }
  
  mid_eta/=denominator;
  mid_phi/=denominator;
  mid_x/=denominator;
  mid_y/=denominator;


  // See = sigma eta eta
  // Spp = (B field corrected) sigma phi phi
  // See = (B field corrected) sigma eta phi
  double See=0.;
  double Spp=0.;
  double Sep=0.;
  double deta(0),dphi(0);
  // compute (phi-corrected) covariance matrix:
  for(unsigned int i=0; i<nCry; i++) {
    if(isBarrel) {
      deta = etaDetId[i]-mid_eta;
      dphi = phiDetId[i]-mid_phi;
    } else {
      deta = etaDetId[i]-mid_eta;
      float hitLocalR2 = (xDetId[i]-mid_x)*(xDetId[i]-mid_x)+(yDetId[i]-mid_y)*(yDetId[i]-mid_y);
      float hitR2 = xDetId[i]*xDetId[i]+yDetId[i]*yDetId[i];
      float meanR2 = mid_x*mid_x+mid_y*mid_y;
      float hitR = sqrt(hitR2);
      float meanR = sqrt(meanR2);
      float phi = acos((hitR2+meanR2-hitLocalR2)/(2*hitR*meanR));
      dphi = hitR*phi;

    }
    See += (wiDetId[i]* deta * deta) / denominator;
    Spp += phiCorrectionFactor*(wiDetId[i]* dphi * dphi) / denominator;
    Sep += sqrt(phiCorrectionFactor)*(wiDetId[i]*deta*dphi) / denominator;
  }

  Cluster2ndMoments returnMoments;

  // compute matrix eigenvalues:
  returnMoments.sMaj = ((See + Spp) + sqrt((See - Spp)*(See - Spp) + 4.*Sep*Sep)) / 2.;
  returnMoments.sMin = ((See + Spp) - sqrt((See - Spp)*(See - Spp) + 4.*Sep*Sep)) / 2.;

  returnMoments.alpha = atan( (See - Spp + sqrt( (Spp - See)*(Spp - See) + 4.*Sep*Sep )) / (2.*Sep));

  return returnMoments;

}




//compute shower shapes: roundness and angle in a vector. Roundness is 0th element, Angle is 1st element.
//description: uses classical mechanics inertia tensor.
//             roundness is smaller_eValue/larger_eValue
//             angle is the angle from the iEta axis to the smallest eVector (a.k.a. the shower's elongated axis)
// this function uses only recHits belonging to a SC above energyThreshold (default 0)
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0) default
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)
std::vector<float> EcalClusterTools::roundnessBarrelSuperClusters( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int weightedPositionMethod, float energyThreshold){//int positionWeightingMethod=0){
    std::vector<const EcalRecHit*>RH_ptrs;
    std::vector< std::pair<DetId, float> > myHitsPair = superCluster.hitsAndFractions();
    std::vector<DetId> usedCrystals;
    for(unsigned int i=0; i< myHitsPair.size(); i++){
        usedCrystals.push_back(myHitsPair[i].first);
    }
    for(unsigned int i=0; i<usedCrystals.size(); i++){
        //get pointer to recHit object
        EcalRecHitCollection::const_iterator myRH = recHits.find(usedCrystals[i]);
        if( myRH != recHits.end() && myRH->energy() > energyThreshold){ //require rec hit to have positive energy
            RH_ptrs.push_back(  &(*myRH)  );
        }
    }
    std::vector<float> temp = EcalClusterTools::roundnessSelectedBarrelRecHits(RH_ptrs,weightedPositionMethod); 
    return temp;
}

// this function uses all recHits within specified window ( with default values ieta_delta=2, iphi_delta=5) around SC's highest recHit.
// recHits must pass an energy threshold "energyRHThresh" (default 0)
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0)
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)

std::vector<float> EcalClusterTools::roundnessBarrelSuperClustersUserExtended( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int ieta_delta, int iphi_delta, float energyRHThresh, int weightedPositionMethod){

    std::vector<const EcalRecHit*>RH_ptrs;
    std::vector< std::pair<DetId, float> > myHitsPair = superCluster.hitsAndFractions();
    std::vector<DetId> usedCrystals;
    for(unsigned int i=0; i< myHitsPair.size(); i++){
        usedCrystals.push_back(myHitsPair[i].first);
    }

    for(unsigned int i=0; i<usedCrystals.size(); i++){
        //get pointer to recHit object
        EcalRecHitCollection::const_iterator myRH = recHits.find(usedCrystals[i]);
        if(myRH != recHits.end() && myRH->energy() > energyRHThresh)
            RH_ptrs.push_back(  &(*myRH)  );
    }


    std::vector<int> seedPosition = EcalClusterTools::getSeedPosition(  RH_ptrs  );

    for(EcalRecHitCollection::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++){
        EBDetId EBdetIdi( rh->detid() );
        //if(rh != recHits.end())
        bool inEtaWindow = (   abs(  deltaIEta(seedPosition[0],EBdetIdi.ieta())  ) <= ieta_delta   );
        bool inPhiWindow = (   abs(  deltaIPhi(seedPosition[1],EBdetIdi.iphi())  ) <= iphi_delta   );
        bool passEThresh = (  rh->energy() > energyRHThresh  );
        bool alreadyCounted = false;

        // figure out if the rechit considered now is already inside the SC
        bool is_SCrh_inside_recHits = false;
        for(unsigned int i=0; i<usedCrystals.size(); i++){
            EcalRecHitCollection::const_iterator SCrh = recHits.find(usedCrystals[i]);
            if(SCrh != recHits.end()){
                is_SCrh_inside_recHits = true;
                if( rh->detid() == SCrh->detid()  ) alreadyCounted = true;
            }
        }//for loop over SC's recHits

        if( is_SCrh_inside_recHits && !alreadyCounted && passEThresh && inEtaWindow && inPhiWindow){
            RH_ptrs.push_back( &(*rh) );
        }

    }//for loop over rh
    return EcalClusterTools::roundnessSelectedBarrelRecHits(RH_ptrs,weightedPositionMethod);
}

// this function computes the roundness and angle variables for vector of pointers to recHits you pass it
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0)
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)
std::vector<float> EcalClusterTools::roundnessSelectedBarrelRecHits( const std::vector<const EcalRecHit*>& RH_ptrs, int weightedPositionMethod){//int weightedPositionMethod = 0){
    //positionWeightingMethod = 0 linear weighting, 1 log energy weighting

    std::vector<float> shapes; // this is the returning vector

    //make sure photon has more than one crystal; else roundness and angle suck
    if(RH_ptrs.size()<2){
        shapes.push_back( -3 );
        shapes.push_back( -3 );
        return shapes;
    }

    //Find highest E RH (Seed) and save info, compute sum total energy used
    std::vector<int> seedPosition = EcalClusterTools::getSeedPosition(  RH_ptrs  );// *recHits);
    int tempInt = seedPosition[0];
    if(tempInt <0) tempInt++;
    float energyTotal = EcalClusterTools::getSumEnergy(  RH_ptrs  );

    //1st loop over rechits: compute new weighted center position in coordinates centered on seed
    float centerIEta = 0.;
    float centerIPhi = 0.;
    float denominator = 0.;

    for(std::vector<const EcalRecHit*>::const_iterator rh_ptr = RH_ptrs.begin(); rh_ptr != RH_ptrs.end(); rh_ptr++){
        //get iEta, iPhi
        EBDetId EBdetIdi( (*rh_ptr)->detid() );
        if(fabs(energyTotal) < 0.0001){
            // don't /0, bad!
            shapes.push_back( -2 );
            shapes.push_back( -2 );
            return shapes;
        }
        float weight = 0;
        if(fabs(weightedPositionMethod)<0.0001){ //linear
            weight = (*rh_ptr)->energy()/energyTotal;
        }else{ //logrithmic
            weight = std::max(0.0, 4.2 + log((*rh_ptr)->energy()/energyTotal));
        }
        denominator += weight;
        centerIEta += weight*deltaIEta(seedPosition[0],EBdetIdi.ieta()); 
        centerIPhi += weight*deltaIPhi(seedPosition[1],EBdetIdi.iphi());
    }
    if(fabs(denominator) < 0.0001){
        // don't /0, bad!
        shapes.push_back( -2 );
        shapes.push_back( -2 );
        return shapes;
    }
    centerIEta = centerIEta / denominator;
    centerIPhi = centerIPhi / denominator;


    //2nd loop over rechits: compute inertia tensor
    TMatrixDSym inertia(2); //initialize 2d inertia tensor
    double inertia00 = 0.;
    double inertia01 = 0.;// = inertia10 b/c matrix should be symmetric
    double inertia11 = 0.;
    int i = 0;
    for(std::vector<const EcalRecHit*>::const_iterator rh_ptr = RH_ptrs.begin(); rh_ptr != RH_ptrs.end(); rh_ptr++){
        //get iEta, iPhi
        EBDetId EBdetIdi( (*rh_ptr)->detid() );

        if(fabs(energyTotal) < 0.0001){
            // don't /0, bad!
            shapes.push_back( -2 );
            shapes.push_back( -2 );
            return shapes;
        }
        float weight = 0;
        if(fabs(weightedPositionMethod) < 0.0001){ //linear
            weight = (*rh_ptr)->energy()/energyTotal;
        }else{ //logrithmic
            weight = std::max(0.0, 4.2 + log((*rh_ptr)->energy()/energyTotal));
        }

        float ieta_rh_to_center = deltaIEta(seedPosition[0],EBdetIdi.ieta()) - centerIEta;
        float iphi_rh_to_center = deltaIPhi(seedPosition[1],EBdetIdi.iphi()) - centerIPhi;

        inertia00 += weight*iphi_rh_to_center*iphi_rh_to_center;
        inertia01 -= weight*iphi_rh_to_center*ieta_rh_to_center;
        inertia11 += weight*ieta_rh_to_center*ieta_rh_to_center;
        i++;
    }

    inertia[0][0] = inertia00;
    inertia[0][1] = inertia01; // use same number here
    inertia[1][0] = inertia01; // and here to insure symmetry
    inertia[1][1] = inertia11;


    //step 1 find principal axes of inertia
    TMatrixD eVectors(2,2);
    TVectorD eValues(2);
    //std::cout<<"EcalClusterTools::showerRoundness- about to compute eVectors"<<std::endl;
    eVectors=inertia.EigenVectors(eValues); //ordered highest eV to lowest eV (I checked!)
    //and eVectors are in columns of matrix! I checked!
    //and they are normalized to 1



    //step 2 select eta component of smaller eVal's eVector
    TVectorD smallerAxis(2);//easiest to spin SC on this axis (smallest eVal)
    smallerAxis[0]=eVectors[0][1];//row,col  //eta component
    smallerAxis[1]=eVectors[1][1];           //phi component

    //step 3 compute interesting quatities
    Double_t temp = fabs(smallerAxis[0]);// closer to 1 ->beamhalo, closer to 0 something else
    if(fabs(eValues[0]) < 0.0001){
        // don't /0, bad!
        shapes.push_back( -2 );
        shapes.push_back( -2 );
        return shapes;
    }

    float Roundness = eValues[1]/eValues[0];
    float Angle=acos(temp);

    if( -0.00001 < Roundness && Roundness < 0) Roundness = 0.;
    if( -0.00001 < Angle && Angle < 0 ) Angle = 0.;

    shapes.push_back( Roundness );
    shapes.push_back( Angle );
    return shapes;

}
//private functions useful for roundnessBarrelSuperClusters etc.
//compute delta iphi between a seed and a particular recHit
//iphi [1,360]
//safe gaurds are put in to ensure the difference is between [-180,180]
int EcalClusterTools::deltaIPhi(int seed_iphi, int rh_iphi){
    int rel_iphi = rh_iphi - seed_iphi;
    // take care of cyclic variable iphi [1,360]
    if(rel_iphi >  180) rel_iphi = rel_iphi - 360;
    if(rel_iphi < -180) rel_iphi = rel_iphi + 360;
    return rel_iphi;
}

//compute delta ieta between a seed and a particular recHit
//ieta [-85,-1] and [1,85]
//safe gaurds are put in to shift the negative ieta +1 to make an ieta=0 so differences are computed correctly
int EcalClusterTools::deltaIEta(int seed_ieta, int rh_ieta){
    // get rid of the fact that there is no ieta=0
    if(seed_ieta < 0) seed_ieta++;
    if(rh_ieta   < 0) rh_ieta++;
    int rel_ieta = rh_ieta - seed_ieta;
    return rel_ieta;
}

//return (ieta,iphi) of highest energy recHit of the recHits passed to this function
std::vector<int> EcalClusterTools::getSeedPosition(const std::vector<const EcalRecHit*>& RH_ptrs){
    std::vector<int> seedPosition;
    float eSeedRH = 0;
    int iEtaSeedRH = 0;
    int iPhiSeedRH = 0;

    for(std::vector<const EcalRecHit*>::const_iterator rh_ptr = RH_ptrs.begin(); rh_ptr != RH_ptrs.end(); rh_ptr++){

        //get iEta, iPhi
        EBDetId EBdetIdi( (*rh_ptr)->detid() );

        if(eSeedRH < (*rh_ptr)->energy()){
            eSeedRH = (*rh_ptr)->energy();
            iEtaSeedRH = EBdetIdi.ieta();
            iPhiSeedRH = EBdetIdi.iphi();
        }

    }// for loop

    seedPosition.push_back(iEtaSeedRH);
    seedPosition.push_back(iPhiSeedRH);
    return seedPosition;
}

// return the total energy of the recHits passed to this function
float EcalClusterTools::getSumEnergy(const std::vector<const EcalRecHit*>& RH_ptrs){

    float sumE = 0.;

    for(std::vector<const EcalRecHit*>::const_iterator rh_ptr = RH_ptrs.begin(); rh_ptr != RH_ptrs.end(); rh_ptr++){
        sumE += (*rh_ptr)->energy();
    }// for loop

    return sumE;
}
