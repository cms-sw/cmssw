#include "RecoHI/HiEgammaAlgos/interface/HiBremRecoveryClusterAlgo.h"
#include "DataFormats/Math/interface/Vector3D.h"
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
        if (verbosity <= pINFO)
        {
           std::cout <<"Basic Cluster: (eta,phi,energy) = "<<cluster_p->position().eta()<<" "<<cluster_p->position().phi()<<" "
        					          <<cluster_p->energy()<<std::endl;
        }

        // if the basic cluster pass the energy threshold -> fill it into the list
        if (fabs(cluster_p->position().eta()) < etaBorder)
        {
           if (cluster_p->energy() > BarrelBremEnergyThreshold) islandClustersBarrel_v.push_back(cluster_p);
        } else {
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

void HiBremRecoveryClusterAlgo::makeIslandSuperClusters(reco::CaloClusterPtrVector &clusters_v, 
						      double etaRoad, double phiRoad)
{
  // Vector of usedSeedEnergy, use the seed energy to check if this cluster is used.
  std::vector<double> usedSeedEnergy;
  usedSeedEnergy.clear();

  // Main brem recovery loop
  for ( reco::CaloCluster_iterator currentSeed = clusters_v.begin(); currentSeed != clusters_v.end(); ++currentSeed)
  {
     if (verbosity <= pINFO) {
    	std::cout <<"Current Cluster: "<<(*currentSeed)->energy()<<" "<<(std::find(usedSeedEnergy.begin(), usedSeedEnergy.end(), (*currentSeed)->energy()) 
    	     != usedSeedEnergy.end())<<std::endl;
     }

     // check this seed was not already used
     if (std::find(usedSeedEnergy.begin(), usedSeedEnergy.end(), (*currentSeed)->energy()) 
    	     != usedSeedEnergy.end()) continue;

     // Does our highest energy cluster have high enough energy? If not, continue instead of break to be robust
     if ((*currentSeed)->energy() * sin((*currentSeed)->position().theta()) < seedTransverseEnergyThreshold) continue;

     // if yes, make it a seed for a new SuperCluster, the position of the SC is calculated by energy weighted sum:
     double energy_ = (*currentSeed)->energy();
     math::XYZVector position_((*currentSeed)->position().X(), 
        		       (*currentSeed)->position().Y(), 
        		       (*currentSeed)->position().Z());
     position_ *= energy_;
     usedSeedEnergy.push_back((*currentSeed)->energy());

     // Printout if verbose
     if (verbosity <= pINFO)
     {
       std::cout << "*****************************" << std::endl;
       std::cout << "******NEW SUPERCLUSTER*******" << std::endl;
       std::cout << "Seed R = " << (*currentSeed)->position().Rho() << std::endl;
       std::cout << "Seed Et = " << (*currentSeed)->energy()* sin((*currentSeed)->position().theta()) << std::endl;
     }

     // and add the matching clusters:
     reco::CaloClusterPtrVector constituentClusters;
     constituentClusters.push_back( *currentSeed );
     reco::CaloCluster_iterator currentCluster = currentSeed + 1;

     // loop over the clusters
     while (currentCluster != clusters_v.end())
     {
        // Print out the basic clusters
        if (verbosity <= pINFO) {
           std::cout <<"->Cluster: "<<(*currentCluster)->energy()
                     <<" Used = "<<(std::find(usedSeedEnergy.begin(), usedSeedEnergy.end(), (*currentCluster)->energy()) != usedSeedEnergy.end())
                     <<" Matched = "<<match(*currentSeed, *currentCluster, etaRoad, phiRoad)<<std::endl;
        }

        // if it's in the search window, and unused
        if (match(*currentSeed, *currentCluster, etaRoad, phiRoad)
            && (std::find(usedSeedEnergy.begin(), usedSeedEnergy.end(), (*currentCluster)->energy()) == usedSeedEnergy.end()))
        {
           // Add basic cluster
           constituentClusters.push_back(*currentCluster);
           energy_   += (*currentCluster)->energy();
           position_ += (*currentCluster)->energy() * math::XYZVector((*currentCluster)->position().X(), 
                                                                      (*currentCluster)->position().Y(), 
                                                                      (*currentCluster)->position().Z()); 
           // Add the cluster to the used list
           usedSeedEnergy.push_back((*currentCluster)->energy());

           if (verbosity <= pINFO) 
           {
              std::cout << "Cluster R = " << (*currentCluster)->position().Rho() << std::endl;
           }

        }
        ++currentCluster;
     }

     // Calculate the final position
     position_ /= energy_;

     if (verbosity <= pINFO)
     {
       std::cout << "Final SuperCluster R = " << position_.Rho() << std::endl;
     }

     // Add the supercluster to the new collection
     reco::SuperCluster newSuperCluster(energy_, 
        				math::XYZPoint(position_.X(), position_.Y(), position_.Z()), 
        				(*currentSeed), 
        				constituentClusters);

     superclusters_v.push_back(newSuperCluster);

     if (verbosity <= pINFO)
       {
         std::cout << "created a new supercluster of: " << std::endl;
         std::cout << "Energy = " << newSuperCluster.energy() << std::endl;
         std::cout << "Position in (R, phi, theta, eta) = (" 
        	   << newSuperCluster.position().Rho() << ", " 
        	   << newSuperCluster.position().phi() << ", "
        	   << newSuperCluster.position().theta() << ", "
        	   << newSuperCluster.position().eta() << ")" << std::endl;
       }
  }
  clusters_v.clear();
  usedSeedEnergy.clear();
}


bool HiBremRecoveryClusterAlgo::match(reco::CaloClusterPtr seed_p, 
				    reco::CaloClusterPtr cluster_p,
				    double dEtaMax, double dPhiMax)
{
  math::XYZPoint clusterPosition = cluster_p->position();
  math::XYZPoint seedPosition = seed_p->position();

  double dPhi = acos(cos(seedPosition.phi() - clusterPosition.phi()));
 
  double dEta = fabs(seedPosition.eta() - clusterPosition.eta());
  if (verbosity <= pINFO) {
     std::cout <<"seed phi: "<<seedPosition.phi()<<" cluster phi: "<<clusterPosition.phi()<<" dphi = "<<dPhi<<" dphiMax = "<<dPhiMax<<std::endl;
     std::cout <<"seed eta: "<<seedPosition.eta()<<" cluster eta: "<<clusterPosition.eta()<<" deta = "<<dEta<<" detaMax = "<<dEtaMax<<std::endl;
  }
  if (dEta > dEtaMax) return false;
  if (dPhi > dPhiMax) return false;

  return true;
}
