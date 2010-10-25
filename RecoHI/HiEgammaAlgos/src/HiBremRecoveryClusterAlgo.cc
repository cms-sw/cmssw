#include "RecoHI/HiEgammaAlgos/interface/HiBremRecoveryClusterAlgo.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

reco::SuperClusterCollection HiBremRecoveryClusterAlgo::makeSuperClusters(reco::CaloClusterPtrVector & clustersCollection)
{
  const float etaBorder = 1.479;

  superclusters_v.clear();
  
  // create vectors of references to clusters of a specific origin...
  reco::CaloClusterPtrVector islandClustersBarrel_v;
  reco::CaloClusterPtrVector islandClustersEndCap_v;

  // ...and populate them:
  for (reco::CaloCluster_iterator it = clustersCollection.begin(); it != clustersCollection.end(); it++)
    {
      reco::CaloClusterPtr cluster_p = *it;
      if (cluster_p->algo() == reco::CaloCluster::island) 
      {
	  if (fabs(cluster_p->position().eta()) < etaBorder)
	    {
	      if (cluster_p->energy() > BarrelBremEnergyThreshold) islandClustersBarrel_v.push_back(cluster_p);
	    }
	  else
	    {
	      if (cluster_p->energy() > EndcapBremEnergyThreshold) islandClustersEndCap_v.push_back(cluster_p);
	    }
      }
    }

  // make the superclusters from the Barrel clusters - Island
  makeIslandSuperClusters(islandClustersBarrel_v, eb_rdeta_, eb_rdphi_);
  // make the superclusters from the EndCap clusters - Island
  makeIslandSuperClusters(islandClustersEndCap_v, ec_rdeta_, ec_rdphi_);

  return superclusters_v;
}

#include "DataFormats/Math/interface/Vector3D.h"

void HiBremRecoveryClusterAlgo::makeIslandSuperClusters(reco::CaloClusterPtrVector &clusters_v, 
						      double etaRoad, double phiRoad)
{
  std::vector<DetId> usedSeedDetIds;
  usedSeedDetIds.clear();

  for ( reco::CaloCluster_iterator currentSeed = clusters_v.begin(); currentSeed != clusters_v.end(); ++currentSeed)
    {

      // check this seed was not already used
      if (std::find(usedSeedDetIds.begin(), usedSeedDetIds.end(), (*currentSeed)->seed()) 
              != usedSeedDetIds.end()) continue;

      // Does our highest energy cluster have high enough energy?
      if ((*currentSeed)->energy() * sin((*currentSeed)->position().theta()) < seedTransverseEnergyThreshold) break;

      // if yes, make it a seed for a new SuperCluster:
      double energy_ = (*currentSeed)->energy();
      math::XYZVector position_((*currentSeed)->position().X(), 
				(*currentSeed)->position().Y(), 
				(*currentSeed)->position().Z());
      position_ *= energy_;

      if (verbosity <= pINFO)
	{
	  std::cout << "*****************************" << std::endl;
	  std::cout << "******NEW SUPERCLUSTER*******" << std::endl;
	  std::cout << "Seed R = " << (*currentSeed)->position().Rho() << std::endl;
	}

      // and add the matching clusters:
      reco::CaloClusterPtrVector constituentClusters;
      constituentClusters.push_back( *currentSeed );
      reco::CaloCluster_iterator currentCluster = currentSeed + 1;
      while (currentCluster != clusters_v.end())
	{
	  if (match(*currentSeed, *currentCluster, etaRoad, phiRoad))
	    {
              // Add basic cluster
	      constituentClusters.push_back(*currentCluster);
	      energy_   += (*currentCluster)->energy();
	      position_ += (*currentCluster)->energy() * math::XYZVector((*currentCluster)->position().X(), 
									 (*currentCluster)->position().Y(), 
									 (*currentCluster)->position().Z()); 
	      // remove cluster from vector of available clusters
	      usedSeedDetIds.push_back((*currentCluster)->seed());

	      if (verbosity <= pINFO) 
		{
		  std::cout << "Cluster R = " << (*currentCluster)->position().Rho() << std::endl;
		}

            }
          ++currentCluster;
	}

      position_ /= energy_;

      if (verbosity <= pINFO)
	{
	  std::cout << "Final SuperCluster R = " << position_.Rho() << std::endl;
	}

      reco::SuperCluster newSuperCluster(energy_, 
					 math::XYZPoint(position_.X(), position_.Y(), position_.Z()), 
					 (*currentSeed), 
					 constituentClusters);

      superclusters_v.push_back(newSuperCluster);

      if (verbosity <= pINFO)
	{
	  std::cout << "created a new supercluster of: " << std::endl;
	  std::cout << "Energy = " << newSuperCluster.energy() << std::endl;
	  std::cout << "Position in (R, phi, theta) = (" 
		    << newSuperCluster.position().Rho() << ", " 
		    << newSuperCluster.position().phi() << ", "
		    << newSuperCluster.position().theta() << ")" << std::endl;
	}
    }
  clusters_v.clear();
}


bool HiBremRecoveryClusterAlgo::match(reco::CaloClusterPtr seed_p, 
				    reco::CaloClusterPtr cluster_p,
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
