
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5ClusterAlgo.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// Return a vector of clusters from a collection of EcalRecHits:
//
std::vector<reco::BasicCluster> Multi5x5ClusterAlgo::makeClusters(
                                  const EcalRecHitCollection* hits,
				  const CaloSubdetectorGeometry *geometry_p,
				  const CaloSubdetectorTopology *topology_p,
				  const CaloSubdetectorGeometry *geometryES_p,
				  EcalPart ecalPart,
				  bool regional,
				  const std::vector<EcalEtaPhiRegion>& regions)
{
  seeds.clear();
  used_s.clear();
  canSeed_s.clear();
  clusters_v.clear();

  recHits_ = hits;

  double threshold = 0;
  std::string ecalPart_string;
  if (ecalPart == endcap) 
    {
      threshold = ecalEndcapSeedThreshold;
      ecalPart_string = "EndCap";
    }
  if (ecalPart == barrel) 
    {
      threshold = ecalBarrelSeedThreshold;
      ecalPart_string = "Barrel";
    }

  if (verbosity < pINFO)
    {
      std::cout << "-------------------------------------------------------------" << std::endl;
      std::cout << "Island algorithm invoked for ECAL" << ecalPart_string << std::endl;
      std::cout << "Looking for seeds, energy threshold used = " << threshold << " GeV" <<std::endl;
    }

  int nregions=0;
  if(regional) nregions=regions.size();

  if(!regional || nregions) {

    EcalRecHitCollection::const_iterator it;
    for(it = hits->begin(); it != hits->end(); it++)
      {
	double energy = it->energy();
	if (energy < threshold) continue; // need to check to see if this line is useful!

	const CaloCellGeometry *thisCell = geometry_p->getGeometry(it->id());
	GlobalPoint position = thisCell->getPosition();

	// Require that RecHit is within clustering region in case
	// of regional reconstruction
	bool withinRegion = false;
	if (regional) {
	  std::vector<EcalEtaPhiRegion>::const_iterator region;
	  for (region=regions.begin(); region!=regions.end(); region++) {
	    if (region->inRegion(position)) {
	      withinRegion =  true;
	      break;
	    }
	  }
	}

	if (!regional || withinRegion) {
	  float ET = it->energy() * sin(position.theta());
	  if (ET > threshold) seeds.push_back(*it);
	}
      }
    
  }
  
   sort(seeds.begin(), seeds.end(), EcalRecHitLess());

   if (verbosity < pINFO)
   {
      std::cout << "Total number of seeds found in event = " << seeds.size() << std::endl;
   }

   mainSearch(hits,geometry_p,topology_p,geometryES_p,ecalPart);
   sort(clusters_v.begin(), clusters_v.end());

   if (verbosity < pINFO)
   {
      std::cout << "---------- end of main search. clusters have been sorted ----" << std::endl;
   }
  
   return clusters_v;
 
}

// Search for clusters
//
void Multi5x5ClusterAlgo::mainSearch(const EcalRecHitCollection* hits,
                                   const CaloSubdetectorGeometry *geometry_p,
                                   const CaloSubdetectorTopology *topology_p,
                                   const CaloSubdetectorGeometry *geometryES_p,
                                   EcalPart ecalPart)
{

   if (verbosity < pINFO)
   {
      std::cout << "Building clusters............" << std::endl;
   }

   // Loop over seeds:
   std::vector<EcalRecHit>::iterator it;
   for (it = seeds.begin(); it != seeds.end(); it++)
   {

      // check if this crystal is able to seed
      // (event though it is already used)
      bool usedButCanSeed = false;
      if (canSeed_s.find(it->id()) != canSeed_s.end()) usedButCanSeed = true;

      // make sure the current seed does not belong to a cluster already.
      if ((used_s.find(it->id()) != used_s.end()) && (usedButCanSeed == false))
      {
         if (it == seeds.begin())
	 {
	    if (verbosity < pINFO)
            {
               std::cout << "##############################################################" << std::endl;
	       std::cout << "DEBUG ALERT: Highest energy seed already belongs to a cluster!" << std::endl;
	       std::cout << "##############################################################" << std::endl;
	    }
	 }

	  // seed crystal is used or is used and cannot seed a cluster
          // so continue to the next seed crystal...
	  continue;
      }

      // clear the vector of hits in current cluster
      current_v.clear();

      // Create a navigator at the seed
      CaloNavigator<DetId> navigator(it->id(), topology_p);
      DetId seedId = navigator.pos();
      navigator.setHome(seedId);

      // Is the seed a local maximum?
      bool localMaxima = checkMaxima(navigator, hits);

      if (localMaxima)
      {
         // build the 5x5 taking care over which crystals
         // can seed new clusters and which can't
         prepareCluster(navigator, hits, geometry_p);
      }

      // If some crystals in the current vector then 
      // make them into a cluster 
      if (current_v.size() > 0) 
      {
         makeCluster(hits, geometry_p, geometryES_p);
      }

   }  // End loop on seed crystals

}

void Multi5x5ClusterAlgo::makeCluster(const EcalRecHitCollection* hits,
				    const CaloSubdetectorGeometry *geometry,
				    const CaloSubdetectorGeometry *geometryES)
{

   double energy = 0;
   double chi2   = 0;
   Point position;
   position = posCalculator_.Calculate_Location(current_v, hits,geometry, geometryES);
  
   std::vector<DetId>::iterator it;
   for (it = current_v.begin(); it != current_v.end(); it++)
   {
      EcalRecHitCollection::const_iterator itt = hits->find(*it);
      EcalRecHit hit_p = *itt;
      energy += hit_p.energy();
      chi2 += 0;
   }
   chi2 /= energy;

   if (verbosity < pINFO)
   { 
      std::cout << "******** NEW CLUSTER ********" << std::endl;
      std::cout << "No. of crystals = " << current_v.size() << std::endl;
      std::cout << "     Energy     = " << energy << std::endl;
      std::cout << "     Phi        = " << position.phi() << std::endl;
      std::cout << "     Eta        = " << position.eta() << std::endl;
      std::cout << "*****************************" << std::endl;
    }

   clusters_v.push_back(reco::BasicCluster(energy, position, chi2, current_v, reco::island));
}

bool Multi5x5ClusterAlgo::checkMaxima(CaloNavigator<DetId> &navigator,
				       const EcalRecHitCollection *hits)
{

   bool maxima = true;
   EcalRecHitCollection::const_iterator thisHit;
   EcalRecHitCollection::const_iterator seedHit = hits->find(navigator.pos());
   double thisEnergy = 0.;
   double seedEnergy = seedHit->energy();

   std::vector<DetId> swissCrossVec;
   swissCrossVec.clear();

   swissCrossVec.push_back(navigator.west());
   navigator.home();
   swissCrossVec.push_back(navigator.east());
   navigator.home();
   swissCrossVec.push_back(navigator.north());
   navigator.home();
   swissCrossVec.push_back(navigator.south());
   navigator.home();

   std::vector<DetId>::const_iterator detItr;
   for (unsigned int i = 0; i < swissCrossVec.size(); ++i)
   {
      thisHit = recHits_->find(swissCrossVec[i]);
      if  ((swissCrossVec[i] == DetId(0)) || thisHit == recHits_->end()) thisEnergy = 0.0;
      else thisEnergy = thisHit->energy();
      if (thisEnergy > seedEnergy)
      {
         maxima = false;
         break;
      }
   }

   return maxima;

}

void Multi5x5ClusterAlgo::prepareCluster(CaloNavigator<DetId> &navigator, 
                const EcalRecHitCollection *hits, 
                const CaloSubdetectorGeometry *geometry)
{

   DetId thisDet;
   std::set<DetId>::iterator setItr;

   // now add the 5x5 taking care to mark the edges
   // as able to seed and where overlapping in the central
   // region with crystals that were previously able to seed
   // change their status so they are not able to seed
   //std::cout << std::endl;
   for (int dx = -2; dx < 3; ++dx)
   {
      for (int dy = -2; dy < 3; ++ dy)
      {

	  // navigate in free steps forming
          // a full 5x5
          thisDet = navigator.offsetBy(dx, dy);
          navigator.home();

          // add the current crystal
	  //std::cout << "adding " << dx << ", " << dy << std::endl;
	  addCrystal(thisDet);

	  // now consider if we are in an edge (outer 16)
          // or central (inner 9) region
          if ((abs(dx) > 1) || (abs(dy) > 1))
          {    
             // this is an "edge" so should be allowed to seed
             // provided it is not already used
             //std::cout << "   setting can seed" << std::endl;
             canSeed_s.insert(thisDet);
          }  // end if "edge"
          else 
          {
             // or else we are in the central 3x3
             // and must remove any of these crystals from the canSeed set
             setItr = canSeed_s.find(thisDet);
             if (setItr != canSeed_s.end())
             {
                //std::cout << "   unsetting can seed" << std::endl;
                canSeed_s.erase(setItr);
             }
          }  // end if "centre"


      } // end loop on dy

   } // end loop on dx

   //std::cout << "*** " << std::endl;
   //std::cout << " current_v contains " << current_v.size() << std::endl;
   //std::cout << "*** " << std::endl;
}


void Multi5x5ClusterAlgo::addCrystal(const DetId &det)
{   

   EcalRecHitCollection::const_iterator thisIt =  recHits_->find(det);
   if ((thisIt != recHits_->end()) && (thisIt->id() != DetId(0)))
   { 
      if ((used_s.find(thisIt->id()) == used_s.end())) 
      {
	 //std::cout << "   ... this is a good crystal and will be added" << std::endl;
         current_v.push_back(det);
         used_s.insert(det);
      }
   } 
  
}

