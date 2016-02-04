#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

reco::SuperClusterCollection Multi5x5BremRecoveryClusterAlgo::makeSuperClusters(reco::CaloClusterPtrVector & clustersCollection)
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
		if (cluster_p->algo() == reco::CaloCluster::multi5x5) 
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

void Multi5x5BremRecoveryClusterAlgo::makeIslandSuperClusters(reco::CaloClusterPtrVector &clusters_v, 
		double etaRoad, double phiRoad)
{

	std::vector<DetId> usedSeedDetIds;
	usedSeedDetIds.clear();

	for (reco::CaloCluster_iterator currentSeed = clusters_v.begin(); currentSeed != clusters_v.end(); ++currentSeed)
	{

		// check this seed was not already used
		if (std::find(usedSeedDetIds.begin(), usedSeedDetIds.end(), (*currentSeed)->seed()) 
			!= usedSeedDetIds.end()) continue;

		// Does our highest energy cluster have high enough energy?
		// changed this to continue from break (to be robust against the order of sorting of the seed clusters)
		if ((*currentSeed)->energy() * sin((*currentSeed)->position().theta()) < seedTransverseEnergyThreshold) continue;

		// if yes, make it a seed for a new SuperCluster:
		double energy = (*currentSeed)->energy();
		math::XYZVector position_((*currentSeed)->position().X(), 
				(*currentSeed)->position().Y(), 
				(*currentSeed)->position().Z());
		position_ *= energy;
		usedSeedDetIds.push_back((*currentSeed)->seed());

LogTrace("EcalClusters") << "*****************************";
LogTrace("EcalClusters") << "******NEW SUPERCLUSTER*******";
LogTrace("EcalClusters") << "Seed R = " << (*currentSeed)->position().Rho();

		reco::CaloClusterPtrVector constituentClusters;
		constituentClusters.push_back(*currentSeed);
		reco::CaloCluster_iterator currentCluster = currentSeed + 1;

		while (currentCluster != clusters_v.end())
		{

			// if dynamic phi road is enabled then compute the phi road for a cluster
			// of energy of existing clusters + the candidate cluster.
			if (dynamicPhiRoad_)
				phiRoad = phiRoadAlgo_->endcapPhiRoad(energy + (*currentCluster)->energy());

			// does the cluster match the phi road for this candidate supercluster
			if (match(*currentSeed, *currentCluster, etaRoad, phiRoad) &&
				std::find(usedSeedDetIds.begin(), usedSeedDetIds.end(), (*currentCluster)->seed()) == usedSeedDetIds.end())
			{

				// add basic cluster to supercluster constituents
				constituentClusters.push_back(*currentCluster);
				energy   += (*currentCluster)->energy();
				position_ += (*currentCluster)->energy() * math::XYZVector((*currentCluster)->position().X(),
						(*currentCluster)->position().Y(), 
						(*currentCluster)->position().Z());

				// remove cluster from vector of available clusters
				usedSeedDetIds.push_back((*currentCluster)->seed());
LogTrace("EcalClusters") << "Cluster R = " << (*currentCluster)->position().Rho();
			}
			++currentCluster;

		}

		position_ /= energy;

LogTrace("EcalClusters") <<"Final SuperCluster R = " << position_.Rho();

		reco::SuperCluster newSuperCluster(energy, 
				math::XYZPoint(position_.X(), position_.Y(), position_.Z()), 
				(*currentSeed), 
				constituentClusters);

		superclusters_v.push_back(newSuperCluster);
LogTrace("EcalClusters") << "created a new supercluster of: ";
LogTrace("EcalClusters") << "Energy = " << newSuperCluster.energy();
LogTrace("EcalClusters") << "Position in (R, phi, theta) = ("
                                << newSuperCluster.position().Rho() << ", "
                                << newSuperCluster.position().phi() << ", "
                                << newSuperCluster.position().theta() << ")" ;


	}

	clusters_v.clear();

}


bool Multi5x5BremRecoveryClusterAlgo::match(reco::CaloClusterPtr seed_p, 
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
