#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

//

// Return a vector of clusters from a collection of EcalRecHits:
std::vector<reco::BasicCluster> IslandClusterAlgo::makeClusters(
                                  const EcalRecHitCollection * hits,
				  const CaloSubdetectorGeometry * geometry_p,
				  const CaloSubdetectorTopology * topology_p,
				  const CaloSubdetectorGeometry * geometryES_p,
				  EcalPart ecalPart,
				  bool regional,
				  const std::vector<EcalEtaPhiRegion> & regions)
{
  seeds.clear();
  used_s.clear();
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
  
  sort(seeds.begin(), seeds.end(), ecalRecHitLess());

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


void IslandClusterAlgo::mainSearch(const EcalRecHitCollection* hits,
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
      // make sure the current seed does not belong to a cluster already.
      if (used_s.find(it->id()) != used_s.end())
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
	  continue;
	}

      // clear the vector of hits in current cluster
      current_v.clear();

      current_v.push_back(it->id());
      used_s.insert(it->id());
      EcalRecHitCollection::const_iterator seedHit_it = recHits_->find(it->id());

      // Create a navigator at the seed
      CaloNavigator<DetId> navigator(it->id(), topology_p);

      searchNorth(navigator, seedHit_it);
      navigator.home();
      searchSouth(navigator, seedHit_it);
      navigator.home();
      searchWest(navigator, seedHit_it);
      navigator.home();
      searchEast(navigator, seedHit_it);
 
      makeCluster(hits,geometry_p,geometryES_p);
   }
}


void IslandClusterAlgo::searchNorth(const CaloNavigator<DetId> & navigator, const EcalRecHitCollection::const_iterator & previous_it)
{
  DetId newId = navigator.north();
  const EcalRecHitCollection::const_iterator newHit_it = iteratorToRecHit(newId);
  if (newHit_it == recHits_->end()) return;

  if (shouldBeAdded(newHit_it, previous_it))
    {
      current_v.push_back(newId);
      used_s.insert(newId);
      searchNorth(navigator, newHit_it);
    }
}


void IslandClusterAlgo::searchSouth(const CaloNavigator<DetId> & navigator, const EcalRecHitCollection::const_iterator & previous_it)
{
  DetId newId = navigator.south();
  const EcalRecHitCollection::const_iterator newHit_it = iteratorToRecHit(newId);
  if (newHit_it == recHits_->end()) return;

  if (shouldBeAdded(newHit_it, previous_it))
    {
      current_v.push_back(newId);
      used_s.insert(newId);
      searchSouth(navigator, newHit_it);
    }
}


void IslandClusterAlgo::searchWest(const CaloNavigator<DetId> & navigator, const EcalRecHitCollection::const_iterator & previous_it)
{
  DetId newId = navigator.west();
  const EcalRecHitCollection::const_iterator newHit_it = iteratorToRecHit(newId);
  if (newHit_it == recHits_->end()) return;

  if (shouldBeAdded(newHit_it, previous_it))
    {
      CaloNavigator<DetId> nsNavigator(newId, navigator.getTopology());

      searchNorth(nsNavigator, newHit_it);
      nsNavigator.home();
      searchSouth(nsNavigator, newHit_it);
      nsNavigator.home();
      searchWest(navigator, newHit_it);

      current_v.push_back(newId);
      used_s.insert(newId);
    }
}


void IslandClusterAlgo::searchEast(const CaloNavigator<DetId> & navigator, const EcalRecHitCollection::const_iterator & previous_it)
{
  DetId newId = navigator.east();
  const EcalRecHitCollection::const_iterator newHit_it = iteratorToRecHit(newId);
  if (newHit_it == recHits_->end()) return;

  if (shouldBeAdded(newHit_it, previous_it))
    {
      CaloNavigator<DetId> nsNavigator(newId, navigator.getTopology());

      searchNorth(nsNavigator, newHit_it);
      nsNavigator.home();
      searchSouth(nsNavigator, newHit_it);
      nsNavigator.home();
      searchEast(navigator, newHit_it);

      current_v.push_back(newId);
      used_s.insert(newId);
    }
}

EcalRecHitCollection::const_iterator IslandClusterAlgo::iteratorToRecHit(const DetId & candidateId)
{
  if (candidateId == DetId(0))                  return (recHits_->end()); // This means that we went off the ECAL!
  if (used_s.find(candidateId) != used_s.end()) return (recHits_->end()); // This means that the added crystal already belongs to another cluster
  return (recHits_->find(candidateId));
}

// returns true if the candidate crystal fulfills the requirements to be added to the cluster:
bool IslandClusterAlgo::shouldBeAdded(const EcalRecHitCollection::const_iterator & candidate_it, 
				      const EcalRecHitCollection::const_iterator & previous_it)
{
  // crystal should not be included...
  if ( (candidate_it->energy() <= 0)                     || // ...if it has a negative or zero energy
       (candidate_it->energy() > previous_it->energy()))    // ...or if the previous crystal had lower E
    {
      return false;
    }
  return true;
}


void IslandClusterAlgo::makeCluster(const EcalRecHitCollection * hits,
				    const CaloSubdetectorGeometry * geometry,
				    const CaloSubdetectorGeometry * geometryES)
{
  double energy = 0;
  double chi2   = 0;

  Point position;
  position = posCalculator_.Calculate_Location(current_v,hits,geometry,geometryES);
  
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
