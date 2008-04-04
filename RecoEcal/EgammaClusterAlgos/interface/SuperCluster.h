#ifndef _RecoECAL_ECALClusters_SuperCluster_h_
#define _RecoECAL_ECALClusters_SuperCluster_h_

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

/*
  Dummy SuperCluster class to be used until we agree on a 
  definition with Luca.
*/

class SuperCluster
{
 private:

  std::vector<reco::BasicCluster *> clusters_v;

 public:

  void add(reco::BasicCluster *newCluster_p)
    {
      clusters_v.push_back(newCluster_p);
    }

  void outputInfo()
    {
      std::cout << "******************************" << std::endl;
      std::cout << "***SUPERCLUSTER INFORMATION***" << std::endl;

      int n = 1;
      std::vector<reco::BasicCluster *>::iterator it;
      for (it = clusters_v.begin(); it != clusters_v.end(); it++)
	{
	  std::cout << "-------Cluster " << n << std::endl;
	  std::cout << "energy = " << (*it)->energy() << std::endl;
	  std::cout << "phi = " << (*it)->position().phi() << std::endl;
	  std::cout << "eta = " << (*it)->position().eta() << std::endl;

	  n++; 
	}

      std::cout << "******************************" << std::endl;
    }

};

#endif 
