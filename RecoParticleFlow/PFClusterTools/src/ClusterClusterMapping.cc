#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

// returns true as soon as the two clusters have one hit in common
bool ClusterClusterMapping::overlap(const reco::CaloCluster & sc1, const reco::CaloCluster & sc2, float minfrac,bool debug)  {
  const std::vector< std::pair<DetId, float> > & hits1 = sc1.hitsAndFractions();
  const std::vector< std::pair<DetId, float> > & hits2 = sc2.hitsAndFractions();
  unsigned nhits1=hits1.size();
  unsigned nhits2=hits2.size();
  
  for(unsigned i1=0;i1<nhits1;++i1)
    { 
      // consider only with a minimum fraction of minfrac (default 1%) of the RecHit
      if(hits1[i1].second<minfrac) {
	  if(debug) std::cout << " Discarding " << hits1[i1].first << " with " << hits1[i1].second << std::endl;
	  continue;
	}
      for(unsigned i2=0;i2<nhits2;++i2)
	{
	  // consider only with a minimum fraction of minfract (default 1%) of the RecHit
	  if(hits2[i2].second<minfrac ) {
	    if(debug) std::cout << " Discarding " << hits2[i2].first << " with " << hits2[i2].second << std::endl;
	    continue;
	  }
	  if(hits1[i1].first==hits2[i2].first)
	    {
	      if(debug)
		{
		  std::cout << " Matching hits " << hits1[i1].first << " with " <<  hits1[i1].second << " and " <<  hits2[i2].first;
		  std::cout << " with " << hits2[i2].second << std::endl;		  
		}
	      return true;
	    }
	}
    }
  return false;
}

bool ClusterClusterMapping::overlap(const reco::PFClusterRef &pfclustest, const reco::SuperCluster &sc, const edm::ValueMap<reco::CaloClusterPtr> &pfclusassoc) {
    
  for (reco::CaloCluster_iterator caloclus = sc.clustersBegin(); caloclus!=sc.clustersEnd(); ++caloclus) {
    const reco::CaloClusterPtr &pfclus = pfclusassoc[*caloclus];
    //printf("pfclus prodidx = %i, key = %i, pfclustest prodidx = %i, key = %i\n",pfclus.id().productIndex(),int(pfclus.key()),pfclustest.id().productIndex(),int(pfclustest.key()));
    if (pfclus.id()==pfclustest.id() && pfclus.key()==pfclustest.key()) return true;    
  }
  return false;
}

int ClusterClusterMapping::checkOverlap(const reco::PFCluster & pfc, const std::vector<const reco::SuperCluster *>& sc,float minfrac,bool debug) {
  int result=-1;
  unsigned nsc=sc.size();
  
  for(unsigned isc=0;isc<nsc;++isc) {
    if(overlap(pfc,*(sc[isc]),minfrac,debug))
      return isc;
  }
  return result;
}

int ClusterClusterMapping::checkOverlap(const reco::PFCluster & pfc, const std::vector<reco::SuperClusterRef >& sc,float minfrac,bool debug) {
  int result=-1;
  unsigned nsc=sc.size();
  
  for(unsigned isc=0;isc<nsc;++isc) {
    if(overlap(pfc,*sc[isc],minfrac,debug))
      return isc;
  }
  return result;
}

int ClusterClusterMapping::checkOverlap(const reco::PFClusterRef & pfc, const std::vector<reco::SuperClusterRef >& sc,const edm::ValueMap<reco::CaloClusterPtr> &pfclusassoc) {
  int result=-1;
  unsigned nsc=sc.size();
  
  for(unsigned isc=0;isc<nsc;++isc) {
    if(overlap(pfc,*sc[isc],pfclusassoc))
      return isc;
  }
  return result;
}
