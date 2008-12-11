#include "RecoEgamma/EgammaHFProducers/interface/HFClusterAlgo.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <list> 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;
using namespace reco;
/** \class HFClusterAlgo
 *
 *  \author Kevin Klapoetke (Minnesota)
 *
 * $Id:version 1.2
 */



HFClusterAlgo::HFClusterAlgo() {
}

class CompareHFCompleteHitET {
public:
  bool operator()(const HFClusterAlgo::HFCompleteHit& h1,const HFClusterAlgo::HFCompleteHit& h2) const {
    return h1.et>h2.et;
  }
};

class CompareHFCore {
public:
  bool operator()(const double c1,const double c2) const {
    return c1>c2;
  }
};

void HFClusterAlgo::setup(double minTowerEnergy) {
  
  m_minTowerEnergy=minTowerEnergy;
}

/** Analyze the hits */
void HFClusterAlgo::clusterize(const HFRecHitCollection& hf, 
			       const CaloGeometry& geom,
			       HFEMClusterShapeCollection& clusterShapes,
			       BasicClusterCollection& BasicClusters,
			       SuperClusterCollection& SuperClusters) {
  
  std::vector<HFCompleteHit> hits, seeds;
  HFRecHitCollection::const_iterator j;
  std::vector<HFCompleteHit>::iterator i;
  std::vector<HFCompleteHit>::iterator k;
  int dP, dE, PWrap;
  bool isok=true;
  HFEMClusterShape clusShp;
  BasicCluster Bclus;
  SuperCluster Sclus;
   bool doCluster=false;
  for (j=hf.begin(); j!= hf.end(); j++)  {
    HFCompleteHit ahit;
    ahit.id=j->id();
    ahit.energy=j->energy();
    double eta=geom.getPosition(j->id()).eta();
    ahit.et=ahit.energy/cosh(eta);
    
    hits.push_back(ahit);
  }
  
  std::sort(hits.begin(), hits.end(), CompareHFCompleteHitET());
  for (i=hits.begin(); i!= hits.end(); i++)  {
    
    if (i->et > 10) {
      if ((i->id.ietaAbs()==40)||(i->id.ietaAbs()==41)||(i->id.ietaAbs()==29)||(i->id.depth()!=1)) 
	isok = false; 
      
      if ( (i==hits.begin()) && (isok) ) {
	doCluster=true;
// 	seeds.push_back(*i);
// 	makeCluster( i->id(),hf, geom,clusShp,Bclus,Sclus);
// 	clusterShapes.push_back(clusShp); 
// 	BasicClusters.push_back(Bclus);
// 	SuperClusters.push_back(Sclus);  	
	//clusterAsoc.insert(SuperClusters,clusterShapes);
  
      }
      else {
	for (k=seeds.begin(); isok && k!=seeds.end(); k++) { //i->hits, k->seeds
	  
	  for (dE=-2; dE<=2; dE++)
	    for (dP=-4;dP<=4; dP+=2) {
	      PWrap=k->id.iphi()+dP;	
	      if (PWrap<0) 
		PWrap+=72;
	      if (PWrap>72)
		PWrap-=72;
	      
	      if ( (i->id.iphi()==PWrap) && (i->id.ieta()==k->id.ieta()+dE))
		isok = false;
	    }
	} 
	if (isok) {
	  doCluster=true;
	}
      }
      if (doCluster) { 
	seeds.push_back(*i);
	makeCluster( i->id(),hf, geom,clusShp,Bclus,Sclus);
	
	clusterShapes.push_back(clusShp);
	BasicClusters.push_back(Bclus);
	SuperClusters.push_back(Sclus);
	
// 	clusterAssoc.insert(edm::Ref<SuperClusterCollection>(SuperClusters,where),
// 			    edm::Ref<HFEMClusterShapeCollection>(clusterShapes,where));
	
      }
      
    }  
    // clusterAsoc.insert(SuperClusters,clusters);
  }
}

void HFClusterAlgo::makeCluster(const HcalDetId& seedid,
				const HFRecHitCollection& hf, 
				const CaloGeometry& geom,
				HFEMClusterShape& clusShp,
				BasicCluster& Bclus,
				SuperCluster& Sclus)  {
			

  double w=0;//sum over all log E's
  double wgt=0;
  double w_e=0;//sum over ieat*energy
  double w_x=0;
  double w_y=0;
  double w_z=0;
  double wp_e=0;//sum over iphi*energy
  double e_e=0;//nonwieghted eta sum
  double e_ep=0; //nonweighted phi sum
  double sum_energy=0;
  double l_3=0;//sum for enenergy in 3x3 long fibers etc.
  double s_3=0;
  double l_5=0;
  double s_5=0;
  double l_1=0;
  double s_1=0;
  int de, dp, ls, phiWrap;
  double l_1e=0;
  GlobalPoint sp=geom.getPosition(seedid);
  std::vector<double> coreCanid;
  std::vector<double>::const_iterator ci;
  HFRecHitCollection::const_iterator i;
  std::vector<DetId> usedHits; 
 
  HFRecHitCollection::const_iterator si;
  HcalDetId sid(HcalForward,seedid.ieta(),seedid.iphi(),1);
  si=hf.find(sid);  

  // lots happens here
  // edge type 1 has 40/41 in 3x3 and 5x5
  bool edge_type1=seedid.ietaAbs()==39 && (seedid.iphi()%4)==3;
  
  for (de=-2; de<=2; de++)
    for (dp=-4;dp<=4; dp+=2) {
      phiWrap=seedid.iphi()+dp;	
      if (phiWrap<0) 
	phiWrap+=72;
      if (phiWrap>72)
	phiWrap-=72;
      
      for (ls=1;ls<=2; ls++){
	
	/* Handling of phi-width change problems */
	if (edge_type1 && de==seedid.zside())
	  if (dp==-2) { // we want it in the 3x3
	    phiWrap-=2;
	    if (phiWrap<0) phiWrap+=72;
	  } else if (dp==-4) continue; // but not double counted in 5x5
	
	HcalDetId id(HcalForward,seedid.ieta()+de,phiWrap,ls);
	i=hf.find(id);
	
	DetId Did(id.rawId());
	usedHits.push_back(Did);
	
	if (i==hf.end()) continue;
	if (i->energy()> m_minTowerEnergy){
	  if (ls==1) {
	    l_5+=i->energy();
	  }
	  
	  if ((ls==1)&&(de>-2)&&(de<2)&&(dp>-4)&&(dp<4)) {
	    l_3+=i->energy();
	  }
	  if ((ls==1)&&(dp==0)&&(de==0)) {
	    l_1=i->energy();
	  }
	  if (ls==2) {
	    s_5+=i->energy();
	  }	  
	  if ((ls==2)&&(de>-2)&&(de<2)&&(dp>-4)&&(dp<4)) {
	    s_3+=i->energy();
	  }
	  if ((ls==2)&&(dp==0)&&(de==0)) {
	    s_1=i->energy();
	  }
	  if ((ls==1)&&(de>-2)&&(de<2)&&(dp>-4)&&(dp<4)&&(i->energy()>(.5*si->energy()))) {
	    coreCanid.push_back(i->energy());
	  }
	  
	  
	  GlobalPoint p=geom.getPosition(id);
	  
	  double d_p = p.phi()-sp.phi();
	  while (d_p < -M_PI)
	    d_p+=2*M_PI;
	  while (d_p > M_PI)
	    d_p-=2*M_PI;
	  double d_e = p.eta()-sp.eta();
	  if((de>-2)&&(de<2)&&(dp>-4)&&(dp<4)/*&&(ls==1)*/ && i->energy()>0) {//long only
	    wgt=log((i->energy()));
	    if (wgt>0){
	      w+=wgt;
	      w_e+=(d_e)*wgt;
	      wp_e+=(d_p)*wgt;
	      e_e+=d_e;
	      e_ep+=d_p;
	      sum_energy+=i->energy();
	      w_x+=(p.x())*wgt;//(p.x()-sp.x())*wgt;
	      w_y+=(p.y())*wgt;
	      w_z+=(p.z())*wgt;
	    }
	  }
	}	
      }
    }
  //Core sorting done here
  std::sort(coreCanid.begin(), coreCanid.end(), CompareHFCore());
  for (ci=coreCanid.begin();ci!=coreCanid.end();ci++){
    if(ci==coreCanid.begin()){
      l_1e=*ci;
    }else if (*ci>.5*l_1e){
      l_1e+=*ci;
    }
  }//core sorting end 
  
  double z_=w_z/w;    //w_z/w+sp.z(); if changed to delta z style
  double x_=w_x/w;
  double y_=w_y/w;
  
  double eta=w_e/w+sp.eta();
  
  double phi=(wp_e/w)+sp.phi();
  
  while (phi < -M_PI)
    phi+=2*M_PI;
  while (phi > M_PI)
    phi-=2*M_PI;
  
  double HFEtaBounds[14] = {2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};
  double RcellEta = eta;
  double Cphi = (phi>0.)?(fmod((phi),0.087*2)/(0.087*2)):((fmod((phi),0.087*2)/(0.087*2))+1.0);
  double Rbin = -1.0;
  for (int icell = 0; icell < 12; icell++ ){
    if ( (RcellEta>HFEtaBounds[icell]) && (RcellEta<HFEtaBounds[icell+1]) )
      Rbin = (RcellEta - HFEtaBounds[icell])/(HFEtaBounds[icell+1] - HFEtaBounds[icell]);
  }
  double Ceta=Rbin;
  
  while (phi< -M_PI)
    phi+=2*M_PI;
  while (phi > M_PI)
    phi-=2*M_PI;
  
  
  math::XYZPoint xyzclus(x_,y_,z_);
  
  double chi2=-1;
  AlgoId algoID = hybrid;
  //return  HFEMClusterShape, BasicCluster, SuperCluster
  
  HFEMClusterShape myClusShp(l_1, s_1, l_3, s_3, l_5,s_5, l_1e,Ceta, Cphi,seedid);
  
  clusShp = myClusShp;
  
  BasicCluster MyBclus(l_3,xyzclus,chi2,usedHits,algoID);
  Bclus=MyBclus;
  
  
  SuperCluster MySclus(l_3,xyzclus);
  Sclus=MySclus;
  
}
