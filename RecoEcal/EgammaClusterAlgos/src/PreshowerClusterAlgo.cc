// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"  
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"  // <===== Still does not exist!
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h" 

#include <vector>
#include <map>


reco::PreshowerCluster PreshowerClusterAlgo::makeOneCluster(ESDetId strip, edm::ESHandle<CaloTopology> theCaloTopology)
{

  int plane = strip.plane();

  std::cout << "Preshower Seeded Algorithm - looking for clusters" << std::endl;
  std::cout << "Preshower is intersected at strip " << strip.strip() << ", at plane " << plane << std::endl;                   

  //  std::vector<EcalRecHit*> dummy; 
  EcalRecHitCollection dummy;
  reco::PreshowerCluster nullcluster=reco::PreshowerCluster(dummy,plane);

  if ( strip == ESDetId(0) ) return nullcluster;
                                                  
  std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator iter_rhits_max;
  
  std::vector<ESDetId> adjacents_east(2);
  std::vector<ESDetId> adjacents_west(2);
  int n = 2*PreshSeededNstr_+1;
  unsigned int m = 3*n;
  std::vector<ESDetId> road_2d(m) ;
  EcalRecHitCollection clusterReco;                                                                                                                                                                                                                                        
  // EcalEndcapNavigator should be replaced by ESNavigator!
  std::cout << "Starting at : (" << strip.six() << "," << strip.siy() << ")" << std::endl;
  //Make a navigator, and set it to the strip cell.
  //  const CaloSubdetectorTopology* esTopology;
  EcalPreshowerNavigator theESNav(strip,theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalPreshower));
  theESNav.setHome(strip);

 //search for neighbours in the road
  ESDetId next;

  road_2d.push_back(strip); 

  if (plane == 1) {
    // east road
    int n_east= 0;
    while ( (next=theESNav.east()) != ESDetId(0) && next != strip && road_2d.size()!=m ) {
       std::cout << "East-0 " << n_east << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       road_2d.push_back(next);   
       ESDetId north_s = theESNav.north();
       std::cout << "East+1 " << n_east << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (north_s != ESDetId(0)) road_2d.push_back(north_s);
       theESNav.south();
       ESDetId south_s = theESNav.south();
       std::cout << "East-1 " << n_east << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (south_s != ESDetId(0)) road_2d.push_back(south_s);
       theESNav.north();
       std::cout << "East-0-test " << n_east << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       ++n_east;  
       if (n_east == PreshSeededNstr_) break; 
    }
    // west road
    int n_west= 0;
    theESNav.home();
    while ( (next=theESNav.west()) != ESDetId(0) && next != strip && road_2d.size()!=m ) {
       std::cout << "West-0 " << n_west << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       road_2d.push_back(next);   
       ESDetId north_s = theESNav.north();
       std::cout << "West+1 " << n_west << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (north_s != ESDetId(0)) road_2d.push_back(north_s);
       theESNav.south();
       ESDetId south_s = theESNav.south();
       std::cout << "West-1 " << n_west << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (south_s != ESDetId(0)) road_2d.push_back(south_s);
       theESNav.north();
       std::cout << "West-0-test " << n_west << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       ++n_west;  
       if (n_west == PreshSeededNstr_) break;     
    }
    std::cout << "Total number of strips found in road is " << n_west+n_east << std::endl;
  } 
  else if (plane == 2) {
    // north road
    int n_north= 0;
    while ( (next=theESNav.north()) != ESDetId(0) && next != strip && road_2d.size()!=m ) {
       std::cout << "North-0 " << n_north << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       road_2d.push_back(next);   
       ESDetId east_s = theESNav.east();
       std::cout << "North+1 " << n_north << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (east_s != ESDetId(0)) road_2d.push_back(east_s);
       theESNav.west();
       ESDetId west_s = theESNav.west();
       std::cout << "North-1 " << n_north << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (west_s != ESDetId(0)) road_2d.push_back(west_s);
       theESNav.east();
       std::cout << "North-0-test " << n_north << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       ++n_north;  
       if (n_north == PreshSeededNstr_) break; 
    }
    // south road
    int n_south= 0;
    theESNav.home();
    while ( (next=theESNav.south()) != ESDetId(0) && next != strip && road_2d.size()!=m ) {
       std::cout << "South-0 " << n_south << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       road_2d.push_back(next);   
       ESDetId east_s = theESNav.east();
       std::cout << "South+1 " << n_south << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if ( east_s != ESDetId(0)) road_2d.push_back(east_s);
       theESNav.south();
       ESDetId west_s = theESNav.west();
       std::cout << "South-1 " << n_south << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       if (west_s != ESDetId(0)) road_2d.push_back(west_s);
       theESNav.east();
       std::cout << "South-0-test " << n_south << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       ++n_south;  
       if (n_south == PreshSeededNstr_) break;     
    }
    std::cout << "Total number of strips found in road is " << n_south+n_north << std::endl;
  } 
  else {
    std::cout << " Wrong plane number, null cluster will be returned! " << std::endl;
    return nullcluster;
  } // end of if

  // Start clustering from strip with max Energy in the road
  float E_max = 0.;
  bool found = false;
  // Loop over strips:
  std::vector<ESDetId>::iterator itID;
  ESDetId strip_max;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
     std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(*itID);
     if ( strip_in_rhits_it == rhits_presh.end() ) {
        std::cout << " There is no such a strip from road_2d among rhits_presh !";
        continue;
     }
     std::pair <EcalRecHit, bool> thisStrip = strip_in_rhits_it->second;
    //If this strip is already used, then don't use it again.
    if (thisStrip.second) continue;
    float E = thisStrip.first.energy();
    std::cout << " This strip of energy E = " <<  E << std::endl;
    if ( E > E_max) {
       E_max = E;
       found = true;
       iter_rhits_max = strip_in_rhits_it;
       strip_max = *itID;
    }
  }

  // no hottest strip found ==> null cluster will be returned
  if ( !found ) return nullcluster;

  // First, save the hottest strip
  //  std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(strip_max);
   std::pair <EcalRecHit, bool> thisStrip = iter_rhits_max->second;
   float E = thisStrip.first.energy();
     // Save strip for clustering if it is not already in use and satisfies energy threshold
   if ( !thisStrip.second && E > PreshStripEnergyCut_) {
      clusterReco.push_back(thisStrip.first);
      std::cout << " Central hottest strip is saved " << std::endl;
      std::cout << " with energy E = " <<  E << std::endl;
   }  

  theESNav.setHome(strip_max);
  if (plane == 1) {
    // Save two neighbouring strips to the east
    int nadjacents_east = 0;
    while ( (next=theESNav.east()) != ESDetId(0) && next != strip && nadjacents_east !=2 ) {
       std::cout << " Adjacent east-0: " << nadjacents_east << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(next);
       std::pair <EcalRecHit, bool> thisStrip = strip_in_rhits_it->second;
       float E = thisStrip.first.energy();
       // Save strip for clustering if it exists and is not already in use, and satisfies energy threshold
       if ( strip_in_rhits_it != rhits_presh.end() && (!thisStrip.second) && E > PreshStripEnergyCut_ ) {
          clusterReco.push_back(thisStrip.first);
          std::cout << " East adjacent strip # " << nadjacents_east << " is saved with energy E = " << E << std::endl;
       }  
       ++nadjacents_east;
    }
    // Save two neighbouring strips to the west
    theESNav.setHome(strip_max);
    int nadjacents_west = 0;
    while ( (next=theESNav.west()) != ESDetId(0) && next != strip && nadjacents_west !=2 ) {
       std::cout << " Adjacent west-0: " << nadjacents_west << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(next);
       std::pair <EcalRecHit, bool> thisStrip = strip_in_rhits_it->second;
       // Save strip for clustering if it exists and is not already in use, and satisfies energy threshold
       if ( strip_in_rhits_it != rhits_presh.end() && (!thisStrip.second)  && E > PreshStripEnergyCut_) {
          clusterReco.push_back(thisStrip.first);
          std::cout << " West adjacent strip # " << nadjacents_west << " is saved with energy E = " << E << std::endl;
       }  
       ++nadjacents_west;
    }
  }
  else if (plane == 2) {
    // Save two neighbouring strips to the north
    int nadjacents_north = 0;
    while ( (next=theESNav.north()) != ESDetId(0) && next != strip && nadjacents_north !=2 ) {
       std::cout << " Adjacent north-0: " << nadjacents_north << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(next);
       std::pair <EcalRecHit, bool> thisStrip = strip_in_rhits_it->second;
       float E = thisStrip.first.energy();
       // Save strip for clustering if it exists and is not already in use, and satisfies energy threshold
       if ( strip_in_rhits_it != rhits_presh.end() && (!thisStrip.second) && E > PreshStripEnergyCut_ ) {
          clusterReco.push_back(thisStrip.first);
          std::cout << " North adjacent strip # " << nadjacents_north << " is saved with energy E = " << E << std::endl;
       }  
       ++nadjacents_north;
    }
    // Save two neighbouring strips to the south
    theESNav.setHome(strip_max);
    int nadjacents_south = 0;
    while ( (next=theESNav.south()) != ESDetId(0) && next != strip && nadjacents_south !=2 ) {
       std::cout << " Adjacent south-0: " << nadjacents_south << " : ("<< next.six() << "," <<   next.siy() << ")" << std::endl;
       std::map<ESDetId, std::pair<EcalRecHit, bool> >::iterator strip_in_rhits_it = rhits_presh.find(next);
       std::pair <EcalRecHit, bool> thisStrip = strip_in_rhits_it->second;
       // Save strip for clustering if it exists and is not already in use, and satisfies energy threshold
       if ( strip_in_rhits_it != rhits_presh.end() && (!thisStrip.second)  && E > PreshStripEnergyCut_) {
          clusterReco.push_back(thisStrip.first);
          std::cout << " South adjacent strip # " << nadjacents_south << " is saved with energy E = " << E << std::endl;
       }  
       ++nadjacents_south;
    }
  }
  else {
    std::cout << " Wrong plane number, null cluster will be returned! " << std::endl;
    return nullcluster;
  } // end of if

  // a cluster is created from vector clusterReco
  reco::PreshowerCluster cluster=reco::PreshowerCluster(clusterReco,plane) ;
  cluster.correct();    // correction method from class PreshowerCluster should exist!

  // return the cluster if its energy is greater a threshold
  if( cluster.Energy() > PreshClusterEnergyCut_ ) 
     return cluster; 
  else
     return nullcluster;

}

void PreshowerClusterAlgo::PreshHitsInit(const EcalRecHitCollection& rhits) 
{
   //Clear the vectors:
   //map that keeps track of used det hits
   rhits_presh.clear();

   std::cout << "Number of Preshower RecHits in event = " << rhits.size() << std::endl;
   EcalRecHitCollection::const_iterator it;

   for (it = rhits.begin(); it != rhits.end(); it++){
     //Double purpose loop:
     //Make the map of DetID, <EcalRecHit,used> pairs and mark hits as unused for clustering ("false")
     std::pair<EcalRecHit, bool> HitBool = std::make_pair(*it, false);
     rhits_presh.insert(std::make_pair(it->id(), HitBool));    
  }

}


