
#include "RecoEcal/EgammaClusterAlgos/interface/CosmicClusterAlgo.h"

#include <vector> //TEMP JHAUPT 4-27

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// Return a vector of clusters from a collection of EcalRecHits:
//
std::vector<reco::BasicCluster> CosmicClusterAlgo::makeClusters(
                                  const EcalRecHitCollection* hits,
				  const CaloSubdetectorGeometry *geometry_p,
				  const CaloSubdetectorTopology *topology_p,
				  const CaloSubdetectorGeometry *geometryES_p,
				  EcalPart ecalPart,
				  const std::vector<int>& masked,
				  bool regional,
				  const std::vector<EcalEtaPhiRegion>& regions)
{
  seeds.clear();
  used_s.clear();
  canSeed_s.clear();
  clusters_v.clear();

  recHits_ = hits;

  //JHAUPT 4-27 TEMP TEMP 
  maskedChannels_.assign(masked.begin(),masked.end());//JHAUPT 4-27 TEMP TEMP
  
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
    //std::cout << "JH12 Seed Energy " << it->energy() << " hashed " << ((EBDetId)it->id()).hashedIndex()  << std::endl;

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
	  //float ET = it->energy() * sin(position.theta()); JHaupt Out 4-27-08 Et not needed for Cosmic Events...
	  if (energy >= threshold) seeds.push_back(*it); // JHaupt 4-27-2008 Et -> energy, most likely not needed as there is already a threshold requirement.
	}
      }
    
  }
  
   sort(seeds.begin(), seeds.end(), EcalRecHitLess());

   if (verbosity < pINFO)
   {
      std::cout << "JH Total number of seeds found in event = " << seeds.size() << std::endl;
	  for (EcalRecHitCollection::const_iterator ji = seeds.begin(); ji != seeds.end(); ++ji)
	  {
	    std::cout << "JH Seed Energy " << ji->energy() << " hashed " << ((EBDetId)ji->id()).hashedIndex()  << std::endl;

	  }
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
void CosmicClusterAlgo::mainSearch(const EcalRecHitCollection* hits,
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
         // build the 5x5 taking care over which crystals //JHaupt 4-27-08 3x3 is a good option...
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

void CosmicClusterAlgo::makeCluster(const EcalRecHitCollection* hits,
				    const CaloSubdetectorGeometry *geometry,
				    const CaloSubdetectorGeometry *geometryES)
{

   double energy = 0;
   double energySecond = 0.;//JHaupt 4-27-08 Added for the second crystal stream
   double energyMax = 0.;//JHaupt 4-27-08 Added for the max crystal stream
   double chi2   = 0;
   EBDetId detFir;
   EBDetId detSec;
   //bool goodCluster = false; //JHaupt 4-27-08 Added so that some can be earased.. used another day Might not be needed as seeds are energy ordered... 
   Point position;
   position = posCalculator_.Calculate_Location(current_v, hits,geometry, geometryES);
   
   std::vector<DetId>::iterator it;
   for (it = current_v.begin(); it != current_v.end(); it++)
   {
      EcalRecHitCollection::const_iterator itt = hits->find(*it);
      EcalRecHit hit_p = *itt;
      if (hit_p.energy() >= 0.027) 
	  {
		energy += hit_p.energy(); //JHaupt only add if greatr than 27 MeV... reevaluate this WARNING!!
		if (hit_p.energy() > energySecond ) {energySecond = hit_p.energy(); detSec = (EBDetId )hit_p.id();}
		if (energySecond > energyMax ) {std::swap(energySecond,energyMax); std::swap(detFir,detSec);}
	  }
      chi2 += 0;
   }
   chi2 /= energy;
   
   if ((energyMax < ecalBarrelSingleThreshold) && (energySecond < ecalBarrelSecondThreshold) ) return;
   
   if (verbosity < pINFO)
   { 
      std::cout << "JH******** NEW CLUSTER ********" << std::endl;
      std::cout << "JHNo. of crystals = " << current_v.size() << std::endl;
      std::cout << "JH     Energy     = " << energy << std::endl;
      std::cout << "JH     Phi        = " << position.phi() << std::endl;
      std::cout << "JH     Eta        = " << position.eta() << std::endl;
      std::cout << "JH*****************************" << std::endl;
      std::cout << "JH****Emax****  "<<energyMax << " ieta " <<detFir.ieta() <<" iphi "<<detFir.ieta()  << std::endl;
      std::cout << "JH****Esec****  "<<energySecond << " ieta " <<detSec.ieta() <<" iphi "<<detSec.ieta() << std::endl;
    }

   clusters_v.push_back(reco::BasicCluster(energy, position, chi2, current_v, reco::island));
}

bool CosmicClusterAlgo::checkMaxima(CaloNavigator<DetId> &navigator,
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

void CosmicClusterAlgo::prepareCluster(CaloNavigator<DetId> &navigator, 
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
   for (int dx = -1; dx < 2; ++dx) //for (int dx = -2; dx < 3; ++dx)
   {
      for (int dy = -1; dy < 2; ++ dy) //for (int dy = -2; dy < 3; ++ dy)
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


void CosmicClusterAlgo::addCrystal(const DetId &det)
{   

   EcalRecHitCollection::const_iterator thisIt =  recHits_->find(det);
   if ((thisIt != recHits_->end()) && (thisIt->id() != DetId(0)))
   { 
      if ((used_s.find(thisIt->id()) == used_s.end())) 
      {
	    used_s.insert(det);
	    if (find(maskedChannels_.begin(), maskedChannels_.end(), ((EBDetId)thisIt->id()).hashedIndex()) == maskedChannels_.end())
	     //std::cout << "   ... this is a good crystal and will be added" << std::endl;
        if (thisIt->energy() >= 0.027)
        {		
		 current_v.push_back(det);
         
		}
      }
   } 
  
}

