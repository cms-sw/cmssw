#include <vector>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TH1.h"

reco::PreshowerCluster PreshowerClusterAlgo::makeOneCluster(ESDetId strip,
							    HitsID *used_strips,
                                                            RecHitsMap *the_rechitsMap_p,
                                                            const CaloSubdetectorGeometry*& geometry_p,
                                                            CaloSubdetectorTopology*& topology_p)
{
  road_2d.clear();

  rechits_map = the_rechitsMap_p;

  used_s = used_strips;

  int plane = strip.plane();
  
  LogTrace("PreShowerClusterAlgo") << "Preshower Seeded Algorithm - looking for clusters";
  LogTrace("PreShowerClusterAlgo")<< "Preshower is intersected at strip" << strip.strip() << ",at plane" << plane ;


  // create null-cluster
  std::vector< std::pair<DetId,float> > dummy;
  Point posi(0,0,0); 
  LogTrace("PreShowerClusterAlgo") <<  " Creating a null-cluster" ;

  reco::PreshowerCluster nullcluster=reco::PreshowerCluster(0.,posi,dummy,plane);

  if ( strip == ESDetId(0) ) return nullcluster;   //works in case of no intersected strip found (e.g. in the Barrel)

  // Collection of cluster strips
  EcalRecHitCollection clusterRecHits;
  // Map of strips for position calculation
  RecHitsMap recHits_pos;

  //Make a navigator, and set it to the strip cell.
  EcalPreshowerNavigator navigator(strip, topology_p);
  navigator.setHome(strip);
 //search for neighbours in the central road
  findRoad(strip,navigator,plane);
  LogTrace("PreShowerClusterAlgo") << "Total number of strips in the central road:" << road_2d.size() ;

  if ( plane == 1 ) {
     ESDetId strip_north = navigator.north();
     findRoad(strip_north,navigator,plane);
     navigator.home();
     ESDetId strip_south = navigator.south();
     findRoad(strip_south,navigator,plane);
     navigator.home();
  }
  if ( plane == 2 ) {
     ESDetId strip_east = navigator.east();
     findRoad(strip_east,navigator,plane);
     navigator.home();
     ESDetId strip_west = navigator.west();
     findRoad(strip_west,navigator,plane);
     navigator.home();
  }
  LogTrace("PreShowerClusterAlgo") << "Total number of strips in all three roads:" << road_2d.size() ;
  
  // Start clustering from strip with max Energy in the road
  float E_max = 0.;
  bool found = false;
  RecHitsMap::iterator max_it;
  // Loop over strips:
  std::vector<ESDetId>::iterator itID;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
    LogTrace("PreShowerClusterAlgo") << "ID ="<<*itID ;
  
    RecHitsMap::iterator strip_it = rechits_map->find(*itID);   
    //if ( strip_it->second.energy() < 0 ) std::cout << "           ##### E = " << strip_it->second.energy() << std::endl;
    if( strip_it==rechits_map->end() || !goodStrip(strip_it)) continue;
    LogTrace("PreShowerClusterAlgo") << " strip is " << ESDetId(strip_it->first) <<"E ="<< strip_it->second.energy();
    
    float E = strip_it->second.energy();
    if ( E > E_max) {
       E_max = E;
       found = true;
       max_it = strip_it;
    }
  }
  // no hottest strip found ==> null cluster will be returned
  if ( !found ) return nullcluster;

  // First, save the hottest strip
  clusterRecHits.push_back(max_it->second);  
  recHits_pos.insert(std::make_pair(max_it->first, max_it->second));
  used_s->insert(max_it->first);
  LogTrace("PreShowerClusterAlgo") << "Central hottest strip" << ESDetId(max_it->first) << "is saved with energy E =" <<  E_max ;
  

  // Find positions of adjacent strips:
  ESDetId next, strip_1, strip_2;
  navigator.setHome(max_it->first);
  ESDetId startES = max_it->first;

   if (plane == 1) {
    // Save two neighbouring strips to the east
    int nadjacents_east = 0;
    while ( (next=navigator.east()) != ESDetId(0) && next != startES && nadjacents_east < 2 ) {
       ++nadjacents_east;
       LogTrace("PreShowerClusterAlgo") << "Adjacent east #" << nadjacents_east <<":"<< next ;
  
       RecHitsMap::iterator strip_it = rechits_map->find(next);

       if(strip_it==rechits_map->end() || !goodStrip(strip_it)) continue;
       // Save strip for clustering if it exists, not already in use, and satisfies an energy threshold
       clusterRecHits.push_back(strip_it->second);       
       // save strip for position calculation
       if ( nadjacents_east==1 ) strip_1 = next;
       used_s->insert(strip_it->first);
       LogTrace("PreShowerClusterAlgo") << "East adjacent strip #" << nadjacents_east << "is saved with energy E =" << strip_it->second.energy() ;
  
    }
    // Save two neighbouring strips to the west
    navigator.home();
    int nadjacents_west = 0;
    while ( (next=navigator.west()) != ESDetId(0) && next != startES && nadjacents_west < 2 ) {
       ++nadjacents_west;
       LogTrace("PreShowerClusterAlgo") << "Adjacent west #" << nadjacents_west <<":"<< next ;
  
       RecHitsMap::iterator strip_it = rechits_map->find(next);
       if(strip_it==rechits_map->end() || !goodStrip(strip_it)) continue;
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_west==1 ) strip_2 = next;
       used_s->insert(strip_it->first);
       LogTrace("PreShowerClusterAlgo") << "West adjacent strip #" << nadjacents_west << "is saved with energy E =" << strip_it->second.energy();
  
    }
  }
  else if (plane == 2) {

  // Save two neighbouring strips to the north
    int nadjacents_north = 0;
    while ( (next=navigator.north()) != ESDetId(0) && next != startES && nadjacents_north < 2 ) {
       ++nadjacents_north;
       LogTrace("PreShowerClusterAlgo") << "Adjacent north #" << nadjacents_north <<":"<< next ;
  
       RecHitsMap::iterator strip_it = rechits_map->find(next); 
       if(strip_it==rechits_map->end() || !goodStrip(strip_it)) continue;      
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_north==1 ) strip_1 = next;
       used_s->insert(strip_it->first);
       LogTrace("PreShowerClusterAlgo") << "North adjacent strip #" << nadjacents_north << "is saved with energy E =" << strip_it->second.energy() ;
       
    }
    // Save two neighbouring strips to the south
    navigator.home();
    int nadjacents_south = 0;
    while ( (next=navigator.south()) != ESDetId(0) && next != startES && nadjacents_south < 2 ) {
       ++nadjacents_south;
       LogTrace("PreShowerClusterAlgo") << "Adjacent south #" << nadjacents_south <<":"<< next ;
       
       RecHitsMap::iterator strip_it = rechits_map->find(next);   
       if(strip_it==rechits_map->end() || !goodStrip(strip_it)) continue;      
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_south==1 ) strip_2 = next;
       used_s->insert(strip_it->first);
       LogTrace("PreShowerClusterAlgo") << "South adjacent strip #" << nadjacents_south << "is saved with energy E =" << strip_it->second.energy() ;
  
    }
  }
  else {
    LogTrace("PreShowerClusterAlgo") << " Wrong plane number" << plane <<", null cluster will be returned! " << std::endl;
    return nullcluster;
  } // end of if

  LogTrace("PreShowerClusterAlgo") << "Total size of clusterRecHits is" << clusterRecHits.size();
  LogTrace("PreShowerClusterAlgo") << "Two adjacent strips for position calculation are:" << strip_1 <<"and" << strip_2;
  

  // strips for position calculation
  RecHitsMap::iterator strip_it1, strip_it2;
  if ( strip_1 != ESDetId(0)) {
    strip_it1 = rechits_map->find(strip_1);
    recHits_pos.insert(std::make_pair(strip_it1->first, strip_it1->second));  
  }
  if ( strip_2 != ESDetId(0) ) {
    strip_it2 = rechits_map->find(strip_2);
    recHits_pos.insert(std::make_pair(strip_it2->first, strip_it2->second));  
  }

  RecHitsMap::iterator cp;
  double energy_pos = 0;
  double x_pos = 0;
  double y_pos = 0;
  double z_pos = 0;
  for (cp = recHits_pos.begin(); cp!=recHits_pos.end(); cp++ ) {
     double E = cp->second.energy();
     energy_pos += E; 
     const CaloCellGeometry *thisCell = geometry_p->getGeometry(cp->first);
     GlobalPoint position = thisCell->getPosition();
     x_pos += E * position.x();
     y_pos += E * position.y();
     z_pos += E * position.z();     
  }
  if(energy_pos>0.) {
     x_pos /= energy_pos;
     y_pos /= energy_pos;
     z_pos /= energy_pos;
  }
  Point pos(x_pos,y_pos,z_pos);
  LogTrace("PreShowerClusterAlgo") << "ES Cluster position =" << "("<< x_pos <<","<< y_pos <<","<< z_pos <<")";

  EcalRecHitCollection::iterator it;
  double Eclust = 0;


  std::vector<std::pair<DetId,float > > usedHits;
  for (it=clusterRecHits.begin(); it != clusterRecHits.end(); it++) {
     Eclust += it->energy();
     usedHits.push_back(std::pair<DetId,float > (it->id(),1.));

  }   
  

  // ES cluster is created from vector clusterRecHits
  reco::PreshowerCluster cluster=reco::PreshowerCluster(Eclust,pos,usedHits,plane);
  LogTrace("PreShowerClusterAlgo") << " ES Cluster is created with:" ;
  LogTrace("PreShowerClusterAlgo") << " energy =" << cluster.energy();
  LogTrace("PreShowerClusterAlgo") << " (eta,phi) =" << "("<<cluster.eta()<<","<<cluster.phi()<<")";
  LogTrace("PreShowerClusterAlgo") << " nhits =" << cluster.nhits();
  LogTrace("PreShowerClusterAlgo") << " radius =" << cluster.position().r();
  LogTrace("PreShowerClusterAlgo") << " (x,y,z) =" << "(" << cluster.x() <<", "<< cluster.y() <<","<< cluster.z()<<")" ;
 
  used_strips = used_s;

  // return the cluster if its energy is greater a threshold
  if( cluster.energy() > preshClusterEnergyCut_ ) 
     return cluster; 
  else
     return nullcluster;

}

// returns true if the candidate strip fulfills the requirements to be added to the cluster:
bool PreshowerClusterAlgo::goodStrip(RecHitsMap::iterator candidate_it)
{
 
  if ( used_s->find(candidate_it->first) != used_s->end())
  LogTrace("PreShowerClusterAlgo") << " This strip is in use";
  if (candidate_it == rechits_map->end() )
  LogTrace("PreShowerClusterAlgo") << " No such a strip in rechits_map";
  if (candidate_it->second.energy() <= preshStripEnergyCut_)
  LogTrace("PreShowerClusterAlgo") << "Strip energy" << candidate_it->second.energy() <<"is below threshold";
  
  // crystal should not be included...
  if ( (used_s->find(candidate_it->first) != used_s->end())  ||       // ...if it already belongs to a cluster
       (candidate_it == rechits_map->end() )                    ||       // ...if it corresponds to a hit
       (candidate_it->second.energy() <= preshStripEnergyCut_ ) )   // ...if it has a negative or zero energy
    {
      return false;
    }
  return true;
}


// find strips in the road of size +/- preshSeededNstr_ from the central strip
void PreshowerClusterAlgo::findRoad(ESDetId strip, EcalPreshowerNavigator theESNav, int plane) {
  
  if ( strip == ESDetId(0) ) return;

   ESDetId next;
   theESNav.setHome(strip);
// First, add a central strip to the road 
   road_2d.push_back(strip);   
   LogTrace("PreShowerClusterAlgo") << "findRoad starts from strip" << strip;
 
   if (plane == 1) {
     // east road
     int n_east= 0;
     LogTrace("PreShowerClusterAlgo") << " Go to the East ";
 
     while ( ((next=theESNav.east()) != ESDetId(0) && next != strip) ) {
       LogTrace("PreShowerClusterAlgo") << "East:" << n_east << "current strip is"<< next;
 
       road_2d.push_back(next);   
       ++n_east;  
       if (n_east == preshSeededNstr_) break; 
     }
     // west road
     int n_west= 0;
     LogTrace("PreShowerClusterAlgo") <<  "Go to the West" ;
    
     theESNav.home();
     while ( ((next=theESNav.west()) != ESDetId(0) && next != strip )) {
       LogTrace("PreShowerClusterAlgo") << "West: " << n_west << "current strip is" << next ;
       
        road_2d.push_back(next);   
        ++n_west;  
        if (n_west == preshSeededNstr_) break; 
     }
     LogTrace("PreShowerClusterAlgo") << "Total number of strips found in the road at 1-st plane is" << n_east+n_west ;
     
  } 
  else if (plane == 2) {
    // north road
    int n_north= 0;
    LogTrace("PreShowerClusterAlgo") << "Go to the North";
    
    while ( ((next=theESNav.north()) != ESDetId(0) && next != strip) ) {   
        LogTrace("PreShowerClusterAlgo") << "North:" << n_north << "current strip is" << next ;
       
       road_2d.push_back(next);   
       ++n_north;  
       if (n_north == preshSeededNstr_) break; 
    }
    // south road
    int n_south= 0;
    LogTrace("PreShowerClusterAlgo") << "Go to the South";
    
    theESNav.home();
    while ( ((next=theESNav.south()) != ESDetId(0) && next != strip) ) {

       LogTrace("PreShowerClusterAlgo") << "South:" << n_south << "current strip is" << next ;
    
       road_2d.push_back(next);   
       ++n_south;  
       if (n_south == preshSeededNstr_) break; 
    }

    LogTrace("PreShowerClusterAlgo") << "Total number of strips found in the road at 2-nd plane is" << n_south+n_north;

  } 
  else {
  LogTrace("PreShowerClusterAlgo") << " Wrong plane number, null cluster will be returned!";

  } // end of if

  theESNav.home();
}



