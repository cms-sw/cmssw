#include "RecoTauTag/HLTProducers/interface/L2TauSimpleClustering.h"
#include "Math/GenVector/VectorUtil.h"


L2TauSimpleClustering::L2TauSimpleClustering()
{
  m_clusterRadius=0.08;
}

L2TauSimpleClustering::L2TauSimpleClustering(double radius)
{
  m_clusterRadius=radius;
}

L2TauSimpleClustering::~L2TauSimpleClustering()
{}


math::PtEtaPhiELorentzVectorCollection 
L2TauSimpleClustering::clusterize(const math::PtEtaPhiELorentzVectorCollection& myRecHits)
{

  math::PtEtaPhiELorentzVectorCollection m_clusters;

   //If we have Hits do Clustering
   if(myRecHits.size()>0)
     {
       //Create the first Cluster by maximum Crystal
       m_clusters.push_back(myRecHits[0]);

       //Loop on The Clusters if there are at least two hits
       if(myRecHits.size()>=2)
       for(math::PtEtaPhiELorentzVectorCollection::const_iterator h = myRecHits.begin()+1;h!=myRecHits.end();++h)
       {
	  //These vars are used to find the nearest clusters to this hits
	  double dR_min=100;  
	  int ptr=0;
	  int ptr_min=-1;
	 
	  for(math::PtEtaPhiELorentzVectorCollection::iterator j=m_clusters.begin()+1;j!=m_clusters.end();j++)
	  {
	     if(ROOT::Math::VectorUtil::DeltaR(*h,*j)<m_clusterRadius)
		{
		  if(ROOT::Math::VectorUtil::DeltaR(*h,*j)<dR_min)
		    {
		      dR_min=ROOT::Math::VectorUtil::DeltaR(*h,*j);
		      ptr_min=ptr;
		    }
		}
		  ptr++;
	  }
	     
	  //If it does not belong to cluster add a new one else add the Crystal to the Cluster
	  if(ptr_min==-1)
	    m_clusters.push_back(*h);
	  else
	    m_clusters[ptr_min]+=*h;
       }
	      
     }
   return m_clusters;
}

