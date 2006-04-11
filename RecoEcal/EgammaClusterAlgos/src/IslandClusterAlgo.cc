#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PositionAwareHit.h"

#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"

#include <iostream>
#include <map>

void IslandClusterAlgo::mainSearch(edm::ESHandle<CaloGeometry> geometry_h)
{
  std::cout << "IslandClusterAlgo Algorithm - looking for clusters" << std::endl;
  std::cout << "Found the following clusters:" << std::endl;
  
  // Loop over seeds:
  std::vector<PositionAwareHit>::iterator it;
  for (it = seeds.begin(); it != seeds.end(); it++)
    {
      // make sure the current seed has not been used/will not be used in the future:
      std::map<EBDetId, PositionAwareHit>::iterator seedInRechits_it = rechits_m.find(it->getId());
      if (seedInRechits_it->second.isUsed()) continue;
      seedInRechits_it->second.use();

      // output some info on the hit:
      std::cout << "*****************************************************" << std::endl;
      std::cout << "Seed of energy E = " << it->getEnergy() 
		<< ", eta = " << it->getEta() 
		<< ", phi = " << it->getPhi() 
		<< std::endl;
      std::cout << "*****************************************************" << std::endl;
      std::cout << "Included RecHits:" << std::endl;      

      reco::EcalRecHitData data(it->getEnergy(),0,it->getId());
      hitData_v.push_back(data);

      EcalBarrelTopology topology(geometry_h);
      const CaloSubdetectorGeometry *geometry_p = (*geometry_h).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      CaloSubdetectorGeometry const geometry = *geometry_p;


      // perform the search:
      EcalBarrelNavigator navigator(it->getId(), &topology);
      searchNorth(navigator);
      navigator.home();
      searchSouth(navigator);
      navigator.home();
      searchWest(navigator, topology);
      navigator.home();
      searchEast(navigator, topology);
      Point pos = getECALposition(hitData_v, geometry);
      clusters.push_back(reco::BasicCluster(hitData_v, 1, pos));
      hitData_v.clear();
    }
}

void IslandClusterAlgo::searchNorth(EcalBarrelNavigator &navigator)
{
  EBDetId southern = navigator.pos();
  std::map<EBDetId, PositionAwareHit>::iterator southern_it = rechits_m.find(southern);
  EBDetId northern = navigator.north();
  std::map<EBDetId, PositionAwareHit>::iterator northern_it = rechits_m.find(northern);

  if (northern_it->second.isUsed()) return;

  if ((northern_it != rechits_m.end()) && (northern_it->second.getEnergy() <= southern_it->second.getEnergy()))
    {
      std::cout << "Going north we find E = " << northern_it->second.getEnergy() << std::endl;

      //      currentCluster->add(northern_it->second.getHit());

      reco::EcalRecHitData data(northern_it->second.getEnergy(), 0, northern_it->second.getId());
      hitData_v.push_back(data);

      searchNorth(navigator);
    }

  northern_it->second.use();
}

void IslandClusterAlgo::searchSouth(EcalBarrelNavigator &navigator)
{
  EBDetId northern = navigator.pos();
  std::map<EBDetId, PositionAwareHit>::iterator northern_it = rechits_m.find(northern);
  EBDetId southern = navigator.south();
  std::map<EBDetId, PositionAwareHit>::iterator southern_it = rechits_m.find(southern);

  if (southern_it->second.isUsed()) return;

  if ((southern_it != rechits_m.end()) && (southern_it->second.getEnergy() <= northern_it->second.getEnergy()))
    {
      std::cout << "Going south we find E = " << southern_it->second.getEnergy() << std::endl;
      //currentCluster->add(southern_it->second.getHit());

      reco::EcalRecHitData data(southern_it->second.getEnergy(), 0, southern_it->second.getId());
      hitData_v.push_back(data);

      searchSouth(navigator);
    }

  southern_it->second.use();
}

void IslandClusterAlgo::searchWest(EcalBarrelNavigator &navigator, EcalBarrelTopology &topology)
{
  EBDetId eastern = navigator.pos();
  std::map<EBDetId, PositionAwareHit>::iterator eastern_it = rechits_m.find(eastern);

  EBDetId western = navigator.west();
  if (western == EBDetId(0)) return; // This means that we went off the barrel!
  std::map<EBDetId, PositionAwareHit>::iterator western_it = rechits_m.find(western);

  if (western_it->second.isUsed()) return;

  EcalBarrelNavigator nsNavigator(western, &topology);

  if ((western_it != rechits_m.end()) && (western_it->second.getEnergy() <= eastern_it->second.getEnergy()))
    {
      std::cout << "Going west we find E = " << western_it->second.getEnergy() << std::endl;
      //currentCluster->add(western_it->second.getHit());

      reco::EcalRecHitData data(western_it->second.getEnergy(), 0, western_it->second.getId());
      hitData_v.push_back(data);

      searchNorth(nsNavigator);
      nsNavigator.home();
      searchSouth(nsNavigator);
      nsNavigator.home();
      searchWest(navigator, topology);
    }

  western_it->second.use();
}

void IslandClusterAlgo::searchEast(EcalBarrelNavigator &navigator, EcalBarrelTopology &topology)
{
  EBDetId western = navigator.pos();
  std::map<EBDetId, PositionAwareHit>::iterator western_it = rechits_m.find(western);

  EBDetId eastern = navigator.east();
  if (eastern == EBDetId(0)) return; // This means that we went off the barrel!
  std::map<EBDetId, PositionAwareHit>::iterator eastern_it = rechits_m.find(eastern);

  if (eastern_it->second.isUsed()) return;

  EcalBarrelNavigator nsNavigator(eastern, &topology);

  if ((eastern_it != rechits_m.end()) && (eastern_it->second.getEnergy() <= western_it->second.getEnergy()))
    {
      std::cout << "Going east we find E = " << eastern_it->second.getEnergy() << std::endl;

      //currentCluster->add(eastern_it->second.getHit());
      reco::EcalRecHitData data(eastern_it->second.getEnergy(), 0, eastern_it->second.getId());
      hitData_v.push_back(data);

      searchNorth(nsNavigator);
      nsNavigator.home();
      searchSouth(nsNavigator);
      nsNavigator.home();
      searchEast(navigator, topology);
    }

  eastern_it->second.use();
}
