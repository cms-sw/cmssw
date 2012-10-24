#include <vector>
#include <map>
#include <iostream>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerPhiClusterAlgo.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "TH1.h"

reco::PreshowerCluster PreshowerPhiClusterAlgo::makeOneCluster(ESDetId strip,
							       HitsID *used_strips,
							       RecHitsMap *the_rechitsMap_p,
							       const CaloSubdetectorGeometry*& geometry_p,
							       CaloSubdetectorTopology*& topology_p,
							       double deltaEta, double minDeltaPhi, double maxDeltaPhi)
{

  rechits_map = the_rechitsMap_p;
  used_s      = used_strips;

  int plane = strip.plane();

  // create null-cluster
  std::vector< std::pair<DetId,float> > dummy;
  Point posi(0,0,0); 
  reco::PreshowerCluster nullcluster = reco::PreshowerCluster(0., posi, dummy, plane);
  if (strip == ESDetId(0)) return nullcluster;   // works in case of no intersected strip found (e.g. in the Barrel)

  const CaloCellGeometry *refCell = geometry_p->getGeometry(strip);
  GlobalPoint refpos = refCell->getPosition();
  double refEta = refpos.eta();
  double refPhi = refpos.phi();  

  // Collection of cluster strips
  EcalRecHitCollection clusterRecHits;

  double x_pos = 0;
  double y_pos = 0;
  double z_pos = 0;

  RecHitsMap::iterator strip_it;
  for (strip_it = rechits_map->begin(); strip_it != rechits_map->end(); ++strip_it) {

    if (!goodStrip(strip_it)) continue; 

    ESDetId mystrip = (strip_it->first == DetId(0)) ? ESDetId(0) : ESDetId(strip_it->first);
    if (mystrip.plane() != plane) continue;

    const CaloCellGeometry *thisCell = geometry_p->getGeometry(strip_it->first);
    GlobalPoint position = thisCell->getPosition();

    if (fabs(position.eta() - refEta) < deltaEta) {

      //std::cout<<"all strips : "<<mystrip.plane()<<" "<<position.phi()<<" "<<reco::deltaPhi(position.phi(), refPhi)<<std::endl;

      if (reco::deltaPhi(position.phi(), refPhi) > 0 && reco::deltaPhi(position.phi(), refPhi) > maxDeltaPhi) continue;
      if (reco::deltaPhi(position.phi(), refPhi) < 0 && reco::deltaPhi(position.phi(), refPhi) < minDeltaPhi) continue;

      //std::cout<<mystrip.zside()<<" "<<mystrip.plane()<<" "<<mystrip.six()<<" "<<mystrip.siy()<<" "<<mystrip.strip()<<" "<<position.eta()<<" "<<position.phi()<<" "<<strip_it->second.energy()<<" "<<strip_it->second.recoFlag()<<" "<<refEta<<" "<<refPhi<<" "<<reco::deltaPhi(position.phi(), refPhi)<<std::endl;

      clusterRecHits.push_back(strip_it->second);
      used_s->insert(strip_it->first);
      
      x_pos += strip_it->second.energy() * position.x();
      y_pos += strip_it->second.energy() * position.y();
      z_pos += strip_it->second.energy() * position.z();     
    }
    
  }

  EcalRecHitCollection::iterator it;
  double Eclust = 0;

  std::vector<std::pair<DetId,float > > usedHits;
  for (it=clusterRecHits.begin(); it != clusterRecHits.end(); it++) {
    Eclust += it->energy();
    usedHits.push_back(std::pair<DetId,float > (it->id(), 1.));
  }   

  if (Eclust > 0.) {
    x_pos /= Eclust;
    y_pos /= Eclust;
    z_pos /= Eclust;
  }
  Point pos(x_pos,y_pos,z_pos);

  reco::PreshowerCluster cluster = reco::PreshowerCluster(Eclust, pos, usedHits, plane);
  used_strips = used_s;

  return cluster; 
}

// returns true if the candidate strip fulfills the requirements to be added to the cluster:
bool PreshowerPhiClusterAlgo::goodStrip(RecHitsMap::iterator candidate_it) {
 
  if ( used_s->find(candidate_it->first) != used_s->end())
  LogTrace("PreShowerPhiClusterAlgo") << " This strip is in use";
  if (candidate_it == rechits_map->end() )
  LogTrace("PreShowerPhiClusterAlgo") << " No such a strip in rechits_map";
  if (candidate_it->second.energy() <= esStripEnergyCut_)
  LogTrace("PreShowerPhiClusterAlgo") << "Strip energy" << candidate_it->second.energy() <<"is below threshold";
  
  // crystal should not be included...
  if ( (used_s->find(candidate_it->first) != used_s->end())  ||       // ...if it already belongs to a cluster
       (candidate_it == rechits_map->end() )                    ||       // ...if it corresponds to a hit
       (candidate_it->second.energy() <= esStripEnergyCut_ ) )   // ...if it has a negative or zero energy
    {
      return false;
    }
  return true;
}

