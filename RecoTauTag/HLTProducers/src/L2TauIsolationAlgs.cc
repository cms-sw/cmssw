#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Math/GenVector/VectorUtil.h"

#include <cmath>
#include <cstdio>

// CLASS IMPLEMENTATION -------------L2TauECALCluster------------------------------

L2TauECALCluster::L2TauECALCluster()
{
  m_etac=0;
  m_phic=0;
  m_et=0;
  m_ncrystals=0;


}

L2TauECALCluster::L2TauECALCluster(const math::PtEtaPhiELorentzVector& c)
{
  m_etac= c.eta();
  m_phic= c.phi();
  m_ncrystals=0;
  m_et=0;

  addCrystal(c);



}



L2TauECALCluster::~L2TauECALCluster()
{}

double 
L2TauECALCluster::etac() const
{return m_etac;}

double
L2TauECALCluster::phic() const
{return m_phic;}

double 
L2TauECALCluster::et() const
{return m_et;}

int
L2TauECALCluster::nCrystals() const
{return m_ncrystals;}



void 
L2TauECALCluster::addCrystal(const math::PtEtaPhiELorentzVector& crystal)
{
  double et,eta,phi;
  et =crystal.pt();
  eta =crystal.eta();
  phi =crystal.phi();
  

  m_ncrystals++;

  m_etac=(m_etac*m_et+eta*et)/(et+m_et);
 
  //Take care of the Phi Boundary when recalcluating Cluster Center

  double tmpphi = fabs(m_phic-phi);
  double d = fabs(tmpphi-2*M_PI);

  if(tmpphi>1.5) //Problem
    {

      if(phi>m_phic) 
	{
	  phi=m_phic-d;
	}
      else
	{
	  phi=m_phic+d;
	}

      m_phic=(m_phic*m_et+phi*et)/(et+m_et);
      
      if(m_phic<-M_PI)
	m_phic=2*M_PI+m_phic;
      if(m_phic>M_PI)
	m_phic=m_phic-2*M_PI;
    }
  else
    {
      m_phic=(m_phic*m_et+phi*et)/(et+m_et);
    
    }

  m_et+=et;
  
 }


//CLASS IMPLEMENTATION L2TauECALClustering-------------------------------------


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
     l2info.ECALClusterNClusters=nClusters();
     l2info.ECALClusterEtaRMS=rms[0];
     l2info.ECALClusterPhiRMS=rms[1];
     l2info.ECALClusterDRRMS=rms[2];
    



  

}


double 
L2TauECALClustering::deltaR(const math::PtEtaPhiELorentzVector& c1,const L2TauECALCluster& c2) const
{



  double deltaEta = fabs(c1.eta()-c2.etac());
  double deltaPhi = fabs(c1.phi()-c2.phic());


  
  
 //look if the crystals are on the phi boundary and fix it
  if(fabs(deltaPhi) > M_PI)
         deltaPhi = fabs( fabs(deltaPhi) - 2*M_PI);

  return sqrt((deltaEta*deltaEta+deltaPhi*deltaPhi));
 
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
	     if(deltaR(*h,*j)<m_clusterRadius)
		{
		  if(deltaR(*h,*j)<dR_min)
		    {
			  dR_min=deltaR(*h,*j);
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
   else
     {

       return;
     }
}


int 
L2TauECALClustering::nClusters() const
{
  return m_clusters.size();
}


std::vector<double> 
L2TauECALClustering::clusterSeperation(const CaloJet& jet) const
{
  double eta_rms=0.;
  double phi_rms=0.;
  double dr_rms=0.;
  double tmpdphi;

  //Get Jet Coordinates
  double jet_eta = jet.eta();
  double jet_phi = jet.phi();


  std::vector<double> rmsVector; //declare the vector
  if(m_clusters.size()>0)
    {
      for(L2TauECALClusterIt c = m_clusters.begin();c!=m_clusters.end();++c) //loop on clusters
	{
	  eta_rms+=pow(c->etac()-jet_eta,2);

	  tmpdphi=c->phic()-jet_phi;
	  if(fabs(tmpdphi) > M_PI)
	    tmpdphi = fabs( fabs(tmpdphi) - 2*M_PI);
	  
	  phi_rms+=pow(tmpdphi,2);
	  dr_rms+=(pow(c->etac()-jet_eta,2)+pow(tmpdphi,2));
	  

	}
         
    }
  else
    {
      eta_rms=0.;
      phi_rms=0.;
      dr_rms =0.;
    }

  rmsVector.push_back(eta_rms);
  rmsVector.push_back(phi_rms);
  rmsVector.push_back(dr_rms);


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

