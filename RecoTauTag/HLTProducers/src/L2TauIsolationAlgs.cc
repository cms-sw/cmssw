#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Math/GenVector/VectorUtil.h"

#include <cmath>
#include <cstdio>

//Class Implementation: L2TauECALCluster

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
  m_threshold = 0.1;
  m_innerCone = 0.2;
  m_outerCone = 0.5;
}

L2TauECALClustering::L2TauECALClustering(double radius,double icone,double ocone,double threshold)
{
  m_clusterRadius=radius;
  m_innerCone = icone;
  m_outerCone = ocone;
  m_threshold = threshold;
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
  l2info.ECALClusterNClusters=nClustersInAnnulus(jet);
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

int 
L2TauECALClustering::nClustersInAnnulus(const CaloJet& jet) const
{
  int clustersInAnnulus = 0;

  for(size_t i = 0; i<m_clusters.size();++i)
    {
      double dr = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),m_clusters[i].p4());
      if(dr<m_outerCone && dr>m_innerCone && m_clusters[i].p4().Pt()>m_threshold)
	 clustersInAnnulus++;
    }

      return clustersInAnnulus;
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
   m_towerEtThreshold=0.5;



}

L2TauTowerIsolation::L2TauTowerIsolation(double threshold,double inner_cone,double outer_cone)
{
   m_innerCone=inner_cone;
   m_outerCone=outer_cone;
   m_towerEtThreshold=threshold;
  


}


L2TauTowerIsolation::~L2TauTowerIsolation()
{

}



void 
L2TauTowerIsolation::run(const CaloJet& jet, L2TauIsolationInfo& l2info)
{
  //Calculate Isolation Energy
  double etIsol = isolatedEt(jet);
  double seedEt = seedTowerEt(jet);
  
  //Fill Trigger Info Class
  l2info.TowerIsolConeCut = etIsol; 
  l2info.SeedTowerEt = seedEt;


 
}


double 
L2TauTowerIsolation::seedTowerEt(const CaloJet& jet) const
{
  //get the sorted calotower collection
  std::vector<CaloTowerPtr> towers = jet.getCaloConstituents();
 
  if(towers.size()>0)
    return (**(towers.begin())).et();
  else
    return 0;


}


double 
L2TauTowerIsolation::isolatedEt(const CaloJet& jet) const
{
  //get the CaloTowers from the jet
  std::vector<CaloTowerPtr> towers = jet.getCaloConstituents();
  
  double eRMin= 0.;
  double eRMax =0.;
  
  for(std::vector<CaloTowerPtr>::iterator u = towers.begin();u!=towers.end();++u)
       if((**u).et()>m_towerEtThreshold)
	{
	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(), (**u).momentum());
	  if(delta<m_outerCone)
	    eRMax+= (**u).et();
	  if(delta<m_innerCone)
	    eRMin+= (**u).et();

	}
    
  double etIsol = eRMax - eRMin;
  
  return etIsol;

}

