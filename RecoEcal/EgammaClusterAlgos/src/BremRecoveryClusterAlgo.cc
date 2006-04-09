#include "RecoEcal/EgammaClusterAlgos/interface/BremRecoveryClusterAlgo.h"

std::vector<SuperCluster> BremRecoveryClusterAlgo::makeSuperClusters(reco::BasicClusterCollection & clustersCollection)
{
  const float etaBorder = 1.479;

  superclusters_v.clear();

  // create vectors of pointers to clusters of a specific origin...
  std::vector<reco::BasicCluster *> islandClustersBarrel_v;
  std::vector<reco::BasicCluster *> islandClustersEndCap_v;
  std::vector<reco::BasicCluster *> hybridClusters_v;

  // ...and populate them:
  reco::BasicClusterCollection::iterator it;
  for (it = clustersCollection.begin(); it != clustersCollection.end(); it++)
    {
      reco::BasicCluster *cluster_p = &(*it);
      if (cluster_p->algo() < 100) 
	{
	  if (cluster_p->position().eta() < etaBorder)
	    {
	      islandClustersBarrel_v.push_back(cluster_p);
	    }
	  else
	    {
	      islandClustersEndCap_v.push_back(cluster_p);
	    }
	}
      else hybridClusters_v.push_back(cluster_p);
    }

  // add the superclusters from the Barrel clusters - Island
  makeIslandSuperClusters(islandClustersBarrel_v, eb_rdeta_, eb_rdphi_);
  // add the superclusters from the EndCap clusters - Island
  makeIslandSuperClusters(islandClustersEndCap_v, ec_rdeta_, ec_rdphi_);
  // add the superclusters from the Hybrid clusters
  makeHybridSuperClusters(hybridClusters_v);
 
  std::cout << "Finished superclustering" << std::endl;

  islandClustersBarrel_v.clear();
  islandClustersEndCap_v.clear();
  hybridClusters_v.clear();

  return superclusters_v;
}


void BremRecoveryClusterAlgo::makeIslandSuperClusters(std::vector<reco::BasicCluster *> &clusters_v, 
					   double etaRoad, double phiRoad)
{
  std::vector<reco::BasicCluster *>::iterator currentSeed;
  for (currentSeed = clusters_v.begin(); !clusters_v.empty(); clusters_v.erase(currentSeed))
    {
      if ((*currentSeed)->energy() < seedEnergyThreshold) break;
      SuperCluster newSuperCluster;
      newSuperCluster.add(*currentSeed);
      
      std::vector<reco::BasicCluster *>::iterator currentCluster = currentSeed + 1;

      while (currentCluster != clusters_v.end())
	{
	  if (match(*currentSeed, *currentCluster, etaRoad, phiRoad))
	    {
	      newSuperCluster.add(*currentCluster);
	      clusters_v.erase(currentCluster);
	    }
	  else currentCluster++;
	}
      newSuperCluster.outputInfo();
      superclusters_v.push_back(newSuperCluster);
    }
  clusters_v.clear();
  currentSeed = clusters_v.end();
}


void BremRecoveryClusterAlgo::makeHybridSuperClusters(std::vector<reco::BasicCluster *> &clusters_v)
{

}


bool BremRecoveryClusterAlgo::match(reco::BasicCluster *seed_p, 
			 reco::BasicCluster *cluster_p,
			 double dEtaMax, double dPhiMax)
{
  math::XYZPoint clusterPosition = cluster_p->position();
  math::XYZPoint seedPosition = seed_p->position();

  double dPhi = acos(cos(seedPosition.phi() - clusterPosition.phi()));
 
  double dEta = fabs(seedPosition.eta() - clusterPosition.eta());
 
  if (dEta > dEtaMax) return false;
  if (dPhi > dPhiMax) return false;

  return true;
}
