#include <vector>
#include <map>

#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "TH1.h"

reco::PreshowerCluster PreshowerClusterAlgo::makeOneCluster(ESDetId strip,
							    HitsID *used_strips,
                                                            RecHitsMap *the_rechitsMap_p,
							    reco::BasicClusterRefVector::iterator basicClust_ref,	     	   
                                                            const CaloSubdetectorGeometry*& geometry_p,
                                                            CaloSubdetectorTopology*& topology_p)
{
  road_2d.clear();

  rechits_map = the_rechitsMap_p;

  used_s = used_strips;

  int plane = strip.plane();

  if ( debugLevel_ <= pINFO ) {
    std::cout << "Preshower Seeded Algorithm - looking for clusters" << std::endl;
    std::cout << "Preshower is intersected at strip " << strip.strip() << ", at plane " << plane << std::endl;
  }

  // create null-cluster
  EcalRecHitCollection dummy;
  Point posi(0,0,0);  
  if ( debugLevel_ <= pINFO ) std::cout << " Creating a null-cluster" << std::endl;
  reco::PreshowerCluster nullcluster=reco::PreshowerCluster(0.,posi,dummy,basicClust_ref,plane);

  if ( strip == ESDetId(0) ) return nullcluster;   //works in case of no intersected strip found (e.g. in the Barrel)

  // Collection of cluster strips
  EcalRecHitCollection clusterRecHits;
  // Map of strips for position calculation
  RecHitsMap recHits_pos;

  // Add to the road the central strip
  road_2d.push_back(strip);

  //Make a navigator, and set it to the strip cell.
  EcalPreshowerNavigator navigator(strip, topology_p);
  navigator.setHome(strip);
 //search for neighbours in the central road
  findRoad(strip,navigator,plane);
  if ( debugLevel_ <= pINFO ) std::cout << "Total number of strips in the central road: " << road_2d.size() << std::endl;

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

  if ( debugLevel_ <= pINFO ) std::cout << "Total number of strips in all three roads: " << road_2d.size() << std::endl;

  // Start clustering from strip with max Energy in the road
  float E_max = 0.;
  bool found = false;
  RecHitsMap::iterator max_it;
  // Loop over strips:
  std::vector<ESDetId>::iterator itID;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
    if ( debugLevel_ == pDEBUG ) std::cout << " ID = " << *itID << std::endl;
    RecHitsMap::iterator strip_it = rechits_map->find(*itID);   
    //if ( strip_it->second.energy() < 0 ) std::cout << "           ##### E = " << strip_it->second.energy() << std::endl;
    if(!goodStrip(strip_it)) continue;
    if ( debugLevel_ == pDEBUG ) std::cout << " strip is " << strip_it->first <<"  E = " << strip_it->second.energy() <<std::endl;
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
  if ( debugLevel_ <= pINFO ) {
     std::cout << " Central hottest strip " << max_it->first << " is saved " << std::endl;
     std::cout << " with energy E = " <<  E_max << std::endl;    
  }

  // Find positions of adjacent strips:
  ESDetId next, strip_1, strip_2;
  navigator.setHome(max_it->first);
  ESDetId startES = max_it->first;

   if (plane == 1) {
    // Save two neighbouring strips to the east
    int nadjacents_east = 0;
    while ( (next=navigator.east()) != ESDetId(0) && next != startES && nadjacents_east < 2 ) {
       ++nadjacents_east;
       if ( debugLevel_ == pDEBUG ) std::cout << " Adjacent east #" << nadjacents_east <<": "<< next << std::endl;
       RecHitsMap::iterator strip_it = rechits_map->find(next);

       if(!goodStrip(strip_it)) continue;
       // Save strip for clustering if it exists, not already in use, and satisfies an energy threshold
       clusterRecHits.push_back(strip_it->second);       
       // save strip for position calculation
       if ( nadjacents_east==1 ) strip_1 = next;
       used_s->insert(strip_it->first);
       if ( debugLevel_ == pDEBUG ) std::cout << " East adjacent strip # " << nadjacents_east << " is saved with energy E = " 
                                              << strip_it->second.energy() << std::endl;             
    }
    // Save two neighbouring strips to the west
    navigator.home();
    int nadjacents_west = 0;
    while ( (next=navigator.west()) != ESDetId(0) && next != startES && nadjacents_west < 2 ) {
       ++nadjacents_west;
       if ( debugLevel_ == pDEBUG ) std::cout << " Adjacent west #" << nadjacents_west <<": "<< next << std::endl; 
       RecHitsMap::iterator strip_it = rechits_map->find(next);
       if(!goodStrip(strip_it)) continue;
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_west==1 ) strip_2 = next;
       used_s->insert(strip_it->first);
       if ( debugLevel_ == pDEBUG ) std::cout << " West adjacent strip # " << nadjacents_west << " is saved with energy E = " 
                                             << strip_it->second.energy() << std::endl;           
    }
  }
  else if (plane == 2) {

  // Save two neighbouring strips to the north
    int nadjacents_north = 0;
    while ( (next=navigator.north()) != ESDetId(0) && next != startES && nadjacents_north < 2 ) {
       ++nadjacents_north;
       if ( debugLevel_ == pDEBUG ) std::cout << " Adjacent north #" << nadjacents_north <<": "<< next << std::endl;   
       RecHitsMap::iterator strip_it = rechits_map->find(next); 
       if(!goodStrip(strip_it)) continue;      
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_north==1 ) strip_1 = next;
       used_s->insert(strip_it->first);
       if ( debugLevel_ == pDEBUG ) std::cout << " North adjacent strip # " << nadjacents_north << " is saved with energy E = " 
                                             << strip_it->second.energy() << std::endl;     
    }
    // Save two neighbouring strips to the south
    navigator.home();
    int nadjacents_south = 0;
    while ( (next=navigator.south()) != ESDetId(0) && next != startES && nadjacents_south < 2 ) {
       ++nadjacents_south;
       if ( debugLevel_ == pDEBUG ) std::cout << " Adjacent south #" << nadjacents_south <<": "<< next << std::endl;   
       RecHitsMap::iterator strip_it = rechits_map->find(next);   
       if(!goodStrip(strip_it)) continue;      
       clusterRecHits.push_back(strip_it->second);
       if ( nadjacents_south==1 ) strip_2 = next;
       used_s->insert(strip_it->first);
       if ( debugLevel_ == pDEBUG ) std::cout << " South adjacent strip # " << nadjacents_south << " is saved with energy E = " 
                                             << strip_it->second.energy() << std::endl;     
    }
  }
  else {
    std::cout << " Wrong plane number" << plane <<", null cluster will be returned! " << std::endl;
    return nullcluster;
  } // end of if
  if ( debugLevel_ <=pINFO ) std::cout << " Total size of clusterRecHits is " << clusterRecHits.size() << std::endl;
  if ( debugLevel_ <=pINFO ) std::cout << " Two adjacent strips for position calculation are: " 
                                      << strip_1 <<" and " << strip_2 << std::endl; 

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
  if ( debugLevel_ == pDEBUG ) std::cout << " ES Cluster position = " << "(" << x_pos <<","<< y_pos <<","<< z_pos <<")"<< std::endl;

  EcalRecHitCollection::iterator it;
  double Eclust = 0;

  if ( debugLevel_ == pINFO ) std::cout << "The found ES cluster strips: " << std::endl;  
  for (it=clusterRecHits.begin(); it != clusterRecHits.end(); it++) {
     Eclust += it->energy();
     if ( debugLevel_ == pINFO ) std::cout << it->id() <<", E = " << it->energy()<<"; ";
  }   
  if ( debugLevel_ == pINFO ) std::cout << std::endl;


  // ES cluster is created from vector clusterRecHits
  reco::PreshowerCluster cluster=reco::PreshowerCluster(Eclust,pos,clusterRecHits,basicClust_ref,plane);

  if ( debugLevel_ <= pINFO ) {
     std::cout << " ES Cluster is created with " << std::endl;
     std::cout << " energy = " << cluster.energy() << std::endl;
     std::cout << " (eta,phi) = " << "("<<cluster.eta()<<", "<<cluster.phi()<<")"<< std::endl;
     std::cout << " nhits = " << cluster.nhits() << std::endl;
     std::cout << " radius = " << cluster.radius() << std::endl; 
     std::cout << " (x,y,z) = " << "(" << cluster.x() <<", "<< cluster.y() <<", "<< cluster.z()<<")"<< std::endl;     
  }
 
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
  if ( debugLevel_ == pDEBUG ) {
    if ( used_s->find(candidate_it->first) != used_s->end()) 
        std::cout << " This strip is in use " << std::endl;    
    if (candidate_it == rechits_map->end() )
        std::cout << " No such a strip in rechits_map " << std::endl; 
    if (candidate_it->second.energy() <= preshStripEnergyCut_)
        std::cout << " Strip energy " << candidate_it->second.energy() <<" is below threshold " << std::endl; 
  }
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

   if ( debugLevel_ <= pINFO ) std::cout << "findRoad starts from strip " << strip << std::endl;  
   if (plane == 1) {
     // east road
     int n_east= 0;
     if ( debugLevel_ == pDEBUG ) std::cout << " Go to the East " <<  std::endl;   
     while ( ((next=theESNav.east()) != ESDetId(0) && next != strip) ) {
        if ( debugLevel_ == pDEBUG ) std::cout << "East: " << n_east << " current strip is " << next << std::endl;  
        road_2d.push_back(next);   
        ++n_east;  
        if (n_east == preshSeededNstr_) break; 
     }
     // west road
     int n_west= 0;
     if ( debugLevel_ == pDEBUG ) std::cout << " Go to the West " <<  std::endl;
     theESNav.home();
     while ( ((next=theESNav.west()) != ESDetId(0) && next != strip )) {
        if ( debugLevel_ == pDEBUG ) std::cout << "West: " << n_west << " current strip is " << next << std::endl;  
        road_2d.push_back(next);   
        ++n_west;  
        if (n_west == preshSeededNstr_) break; 
     }
     if ( debugLevel_ == pDEBUG ) std::cout << "Total number of strips found in the road at 1-st plane is " << n_east+n_west << std::endl;
  } 
  else if (plane == 2) {
    // north road
    int n_north= 0;
    if ( debugLevel_ == pDEBUG ) std::cout << " Go to the North " <<  std::endl;
    while ( ((next=theESNav.north()) != ESDetId(0) && next != strip) ) {       
       if ( debugLevel_ == pDEBUG ) std::cout << "North: " << n_north << " current strip is " << next << std::endl; 
       road_2d.push_back(next);   
       ++n_north;  
       if (n_north == preshSeededNstr_) break; 
    }
    // south road
    int n_south= 0;
    if ( debugLevel_ == pDEBUG ) std::cout << " Go to the South " <<  std::endl;
    theESNav.home();
    while ( ((next=theESNav.south()) != ESDetId(0) && next != strip) ) {
       if ( debugLevel_ == pDEBUG ) std::cout << "South: " << n_south << " current strip is " << next << std::endl;      
       road_2d.push_back(next);   
       ++n_south;  
       if (n_south == preshSeededNstr_) break; 
    }
    if ( debugLevel_ == pDEBUG ) std::cout << "Total number of strips found in the road at 2-nd plane is " << n_south+n_north << std::endl;
  } 
  else {
    if ( debugLevel_ == pDEBUG ) std::cout << " Wrong plane number, null cluster will be returned! " << std::endl;    
  } // end of if

  theESNav.home();
}



