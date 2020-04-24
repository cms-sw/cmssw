#include "DataFormats/Common/interface/Handle.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

L1Analysis::L1AnalysisRecoCluster::L1AnalysisRecoCluster()
{
}


L1Analysis::L1AnalysisRecoCluster::~L1AnalysisRecoCluster()
{
}

void L1Analysis::L1AnalysisRecoCluster::Set(const reco::CaloClusterCollection & caloClusters, unsigned maxCl)
{
  recoCluster_.nClusters=recoCluster_.eta.size();
  for(reco::CaloClusterCollection::const_iterator it=caloClusters.begin(); it!=caloClusters.end() && recoCluster_.nClusters<maxCl; it++)
  {
     recoCluster_.eta.push_back( it->eta() );
     recoCluster_.phi.push_back( it->phi() );
     recoCluster_.et.push_back ( it->energy() * sin( it->position().theta() ) );
     recoCluster_.e.push_back  ( it->energy() );
     recoCluster_.nClusters++;
  }

}

void L1Analysis::L1AnalysisRecoCluster::Set(const reco::SuperClusterCollection & superClusters, unsigned maxCl)
{
   recoCluster_.nClusters=recoCluster_.eta.size();

  for(reco::SuperClusterCollection::const_iterator it=superClusters.begin(); it!=superClusters.end() && recoCluster_.nClusters<maxCl; it++)
  {
     recoCluster_.eta.push_back(it->eta());
     recoCluster_.phi.push_back(it->phi());
     recoCluster_.et.push_back (it->energy() * sin( it->position().theta() ) );
     recoCluster_.e.push_back  (it->energy());
     recoCluster_.nClusters++;
   }

}



