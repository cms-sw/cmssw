#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"

reco::SuperClusterCollection Multi5x5BremRecoveryClusterAlgo::makeSuperClusters(reco::BasicClusterRefVector & clustersCollection)
{
  const float etaBorder = 1.479;

  superclusters_v.clear();
  
  // create vectors of references to clusters of a specific origin...
  reco::BasicClusterRefVector islandClustersBarrel_v;
  reco::BasicClusterRefVector islandClustersEndCap_v;

  // ...and populate them:
  reco::basicCluster_iterator it;
  for (it = clustersCollection.begin(); it != clustersCollection.end(); it++)
    {
      reco::BasicClusterRef cluster_p = *it;
      if (cluster_p->algo() == reco::island) 
      {
	  if (fabs(cluster_p->position().eta()) < etaBorder)
	    {
	      islandClustersBarrel_v.push_back(cluster_p);
	    }
	  else
	    {
	      islandClustersEndCap_v.push_back(cluster_p);
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

void Multi5x5BremRecoveryClusterAlgo::makeIslandSuperClusters(reco::BasicClusterRefVector &clusters_v, 
						      double etaRoad, double phiRoad)
{

  reco::basicCluster_iterator currentSeed;
  for (currentSeed = clusters_v.begin(); !clusters_v.empty(); clusters_v.erase(currentSeed))
    {
      // Does our highest energy cluster have high enough energy?
      if ((*currentSeed)->energy() * sin((*currentSeed)->position().theta()) < seedTransverseEnergyThreshold) break;

      // if yes, make it a seed for a new SuperCluster:
      double energy = (*currentSeed)->energy();
      math::XYZVector position_((*currentSeed)->position().X(), 
				(*currentSeed)->position().Y(), 
				(*currentSeed)->position().Z());
      position_ *= energy;


      if (verbosity <= pINFO)
	{
	  std::cout << "*****************************" << std::endl;
	  std::cout << "******NEW SUPERCLUSTER*******" << std::endl;
	  std::cout << "Seed R = " << (*currentSeed)->position().Rho() << std::endl;
	}

      // and add the matching clusters:
      reco::BasicClusterRefVector constituentClusters;
      constituentClusters.push_back(*currentSeed);
      reco::basicCluster_iterator currentCluster = currentSeed + 1;
      while (currentCluster != clusters_v.end())
	{

	  // if dynamic phi road is enabled then compute the phi road for a cluster
 	  // of energy of existing clusters + the candidate cluster.
	  if (dynamicPhiRoad_) {
             phiRoad = phiRoadAlgo_->endcapPhiRoad(energy + (*currentCluster)->energy());
          }

	  if (match(*currentSeed, *currentCluster, etaRoad, phiRoad))
	    {
	      constituentClusters.push_back(*currentCluster);
	      energy   += (*currentCluster)->energy();
	      position_ += (*currentCluster)->energy() * math::XYZVector((*currentCluster)->position().X(), 
									 (*currentCluster)->position().Y(), 
									 (*currentCluster)->position().Z()); 
	      if (verbosity <= pINFO) 
		{
		  std::cout << "Cluster R = " << (*currentCluster)->position().Rho() << std::endl;
		}

	      clusters_v.erase(currentCluster);
	    }
	  else currentCluster++;
	}

      position_ /= energy;

      if (verbosity <= pINFO)
	{
	  std::cout << "Final SuperCluster R = " << position_.Rho() << std::endl;
	}

      reco::SuperCluster newSuperCluster(energy, 
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
  currentSeed = clusters_v.end();
}


bool Multi5x5BremRecoveryClusterAlgo::match(reco::BasicClusterRef seed_p, 
				    reco::BasicClusterRef cluster_p,
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
