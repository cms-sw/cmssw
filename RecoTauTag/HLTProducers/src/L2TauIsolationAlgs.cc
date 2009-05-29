#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Math/GenVector/VectorUtil.h"

#include <cmath>
#include <cstdio>

//Class Implementation: L2TauECALCluster

using namespace reco;

L2TauECALCluster::L2TauECALCluster()
{
  p4_ = math::PtEtaPhiELorentzVector();
  m_ncrystals=0;
}

L2TauECALCluster::L2TauECALCluster(const math::PtEtaPhiELorentzVector& c)
{
  m_ncrystals=0;
  p4_ = math::PtEtaPhiELorentzVector();
  addCrystal(c);
}



L2TauECALCluster::~L2TauECALCluster()
{}


int
L2TauECALCluster::nCrystals() const
{return m_ncrystals;}

math::PtEtaPhiELorentzVector
L2TauECALCluster::p4() const
{return p4_;}


void 
L2TauECALCluster::addCrystal(const math::PtEtaPhiELorentzVector& crystal)
{
  p4_+=crystal;
}


//Class Implementation: ECAL Clustering


L2TauECALClustering::L2TauECALClustering()
{
  m_clusterRadius=0.08;
}

L2TauECALClustering::L2TauECALClustering(double radius)
{
  m_clusterRadius=radius;
}

L2TauECALClustering::~L2TauECALClustering()
{}

void
L2TauECALClustering::run(const math::PtEtaPhiELorentzVectorCollection& hits,const CaloJet& jet,L2TauIsolationInfo& l2info)
{
  //Create Clusters
  clusterize(hits);


  //Fill info Class
  std::vector<double> rms  = clusterSeperation(jet);
  l2info.ECALClusterNClusters=m_clusters.size();
  l2info.ECALClusterEtaRMS=rms[0];
  l2info.ECALClusterPhiRMS=rms[1];
  l2info.ECALClusterDRRMS=rms[2];
}

void 
L2TauECALClustering::clusterize(const math::PtEtaPhiELorentzVectorCollection& myRecHits)
{
   //If we have Hits do Clustering
   if(myRecHits.size()>0)
     {
       //Create the first Cluster by maximum Crystal
       m_clusters.push_back(L2TauECALCluster(*(myRecHits.begin())));

       //Loop on The Clusters if there are at least two hits
       if(myRecHits.size()>=2)
       for(math::PtEtaPhiELorentzVectorCollection::const_iterator h = myRecHits.begin()+1;h!=myRecHits.end();++h)
       {
	  //These vars are used to find the nearest clusters to this hits
	  double dR_min=100;  
	  int ptr=0;
	  int ptr_min=-1;
	 
	  for(L2TauECALClusterIt j=m_clusters.begin()+1;j!=m_clusters.end();j++)
	  {
	     if(ROOT::Math::VectorUtil::DeltaR(*h,j->p4())<m_clusterRadius)
		{
		  if(ROOT::Math::VectorUtil::DeltaR(*h,j->p4())<dR_min)
		    {
		      dR_min=ROOT::Math::VectorUtil::DeltaR(*h,j->p4());
			  ptr_min=ptr;
		    }
		}
		  ptr++;
	  }
	     
	  //If it does not belong to cluster add a new one else add the Crystal to the Cluster
	  if(ptr_min==-1)
	    m_clusters.push_back(L2TauECALCluster(*h));
	  else
	    m_clusters[ptr_min].addCrystal(*h);

       }
	      
     }

}


std::vector<double> 
L2TauECALClustering::clusterSeperation(const CaloJet& jet) const
{
  double eta_rms=0.;
  double phi_rms=0.;
  double dr_rms=0.;

  double sumpt = 0.;

  std::vector<double> rmsVector; //declare the vector
  if(m_clusters.size()>0)
    {
      for(L2TauECALClusterIt c = m_clusters.begin();c!=m_clusters.end();++c) //loop on clusters
	{
	  eta_rms+=c->p4().Pt()*pow(c->p4().Eta()-jet.eta(),2);
	  phi_rms+=c->p4().Pt()*pow(ROOT::Math::VectorUtil::DeltaPhi(c->p4(),jet.p4().Vect()),2);
	  dr_rms+=c->p4().Pt()*pow(ROOT::Math::VectorUtil::DeltaR(c->p4(),jet.p4().Vect()),2);
	  sumpt+=c->p4().Pt();			   
	}
    }
  else
    {
      eta_rms=0.;
      phi_rms=0.;
      dr_rms =0.;
      sumpt=1.;
    }

  rmsVector.push_back(eta_rms/sumpt);
  rmsVector.push_back(phi_rms/sumpt);
  rmsVector.push_back(dr_rms/sumpt);

  return rmsVector;
}


//CLASS IMPLEMENTATION ----------------ECAL Isolation-------------------------

L2TauECALIsolation::L2TauECALIsolation()
{
   m_innerCone=0.15;
   m_outerCone=0.50;
}

L2TauECALIsolation::L2TauECALIsolation(double inner_cone,double outer_cone)
{
   m_innerCone=inner_cone;
   m_outerCone=outer_cone;
}



L2TauECALIsolation::~L2TauECALIsolation()
{

}

void 
L2TauECALIsolation::run(const  math::PtEtaPhiELorentzVectorCollection& hits,const CaloJet& jet,L2TauIsolationInfo& l2info)
{
  //Calculate Isolation Energy
  double etIsol = isolatedEt(hits,jet);
  
  
  //Fill Trigger Info Class
  l2info.ECALIsolConeCut = etIsol; 


 }

double 
L2TauECALIsolation::isolatedEt(const math::PtEtaPhiELorentzVectorCollection& hits,const CaloJet& jet) const
{
  //Compute the Isolation
  

  double eRMax = 0.;
  double eRMin = 0;

  for (math::PtEtaPhiELorentzVectorCollection::const_iterator mRH = hits.begin();mRH!=hits.end();++mRH)
    {
	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),(*mRH));
	  if(delta <m_outerCone)
	    eRMax+= mRH->pt();
	  if(delta <m_innerCone)
	    eRMin+= mRH->pt();
    }

  double etIsol = eRMax - eRMin;


  return etIsol;

}


//CLASS IMPLEMENTATION --------------------L2TauTowerIsolation


L2TauTowerIsolation::L2TauTowerIsolation()
{
   m_innerCone=0.20;
   m_outerCone=0.50;




}

L2TauTowerIsolation::L2TauTowerIsolation(double inner_cone,double outer_cone)
{
   m_innerCone=inner_cone;
   m_outerCone=outer_cone;

  


}


L2TauTowerIsolation::~L2TauTowerIsolation()
{

}



void 
L2TauTowerIsolation::run(const CaloJet& jet,const math::PtEtaPhiELorentzVectorCollection& towers,L2TauIsolationInfo& l2info)
{
  //Calculate Isolation Energy
  double etIsol = isolatedEt(jet,towers);
  double seedEt = seedTowerEt(towers);
  
  //Fill Trigger Info Class
  l2info.TowerIsolConeCut = etIsol; 
  l2info.SeedTowerEt = seedEt;


 
}


double 
L2TauTowerIsolation::seedTowerEt(const math::PtEtaPhiELorentzVectorCollection& towers) const
{
  //get the sorted calotower collection
 
  if(towers.size()>0)
    return (towers[0].pt());
  else
    return 0;


}


double 
L2TauTowerIsolation::isolatedEt(const CaloJet& jet,const math::PtEtaPhiELorentzVectorCollection& towers ) const
{
  
  double eRMin= 0.;
  double eRMax =0.;
  
  for(math::PtEtaPhiELorentzVectorCollection::const_iterator u = towers.begin();u!=towers.end();++u)
	{
	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4(), *u);
	  if(delta<m_outerCone)
	    eRMax+=u->pt();
	  if(delta<m_innerCone)
	    eRMin+= u->pt();
	}
    
  double etIsol = eRMax - eRMin;
  return etIsol;
}

