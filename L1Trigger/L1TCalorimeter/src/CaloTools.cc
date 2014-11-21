#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

const l1t::CaloTower l1t::CaloTools::nullTower_;
const l1t::CaloCluster l1t::CaloTools::nullCluster_;

//currently implemented as a brute force search but this will hopefully change in the future
//with standarising the layout of std::vector<l1t::CaloTower>
const l1t::CaloTower& l1t::CaloTools::getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi)
{
  size_t towerIndex = CaloTools::caloTowerHash(iEta, iPhi);
  if(towerIndex<towers.size()){
    if(towers[towerIndex].hwEta()!=iEta || towers[towerIndex].hwPhi()!=iPhi){ //it failed, this is bad, but we will not log the error due to policy and silently attempt to do a brute force search instead 
      // std::cout <<"error, tower "<<towers[towerIndex].hwEta()<<" "<<towers[towerIndex].hwPhi()<<" does not match "<<iEta<<" "<<iPhi<<" index "<<towerIndex<<" nr towrs "<<towers.size()<<std::endl;
      for(size_t towerNr=0;towerNr<towers.size();towerNr++){
	if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
      }     
    }else return towers[towerIndex];
  
  }
  else{// in case the vector of towers do not contain all the towers (towerIndex can be > towers.size())
    for(size_t towerNr=0;towerNr<towers.size();towerNr++){
	  if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
    }
  }

  return nullTower_;
}

const l1t::CaloCluster& l1t::CaloTools::getCluster(const std::vector<l1t::CaloCluster>& clusters,int iEta,int iPhi)
{
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    if(clusters[clusterNr].hwEta()==iEta && clusters[clusterNr].hwPhi()==iPhi) return clusters[clusterNr];
  }
  return nullCluster_;
}



//this implimentation has not all the necessary info yet, we need to check the exact HF numbering
//(iEta=-28,iPhi=1)=index 0 to (iEta=28,iPhi=72)=index 28*72*2-1
//HF then runs after that so -32,1 = 28*72*2
size_t l1t::CaloTools::caloTowerHash(int iEta,int iPhi)
{

  if(!isValidIEtaIPhi(iEta,iPhi)) return caloTowerHashMax();
  else{
    const int absIEta = abs(iEta);
    if(absIEta>kHFEnd) return kNrTowers;
    else if(absIEta<=kHBHEEnd){ //HBHE
      int iEtaNoZero=iEta;
      if(iEta>0) iEtaNoZero--;
      return (iEtaNoZero+kHBHEEnd)*kHBHENrPhi+iPhi-1;
    }else{ //HF
      int iEtaIndex = iEta+kHFEnd; //iEta=-32 is 0
      if(iEta>0) iEtaIndex= iEta-kHBHEEnd+(kHFEnd-kHBHEEnd)-1; //but iEta=29 is 4
      return iEtaIndex*kHFNrPhi+iPhi/kHFPhiSeg + kNrHBHETowers;
    }
  }
}


size_t l1t::CaloTools::caloTowerHashMax()
{
  return kNrTowers;
}


bool l1t::CaloTools::isValidIEtaIPhi(int iEta,int iPhi)
{
  size_t absIEta = abs(iEta);
  if(iPhi<=0 || iPhi>kHBHENrPhi) return false;
  if(absIEta==0 || absIEta>kHFEnd) return false;
  if(absIEta>kHBHEEnd && iPhi%kHFPhiSeg!=1) return false;
  return true;

}

int l1t::CaloTools::calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			       int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,
			       SubDet etMode)
{

  return calHwEtSum(iEta,iPhi,towers,localEtaMin,localEtaMax,localPhiMin,localPhiMax,kHFEnd,etMode);
}

int l1t::CaloTools::calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			       int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,
			       int iEtaAbsMax,SubDet etMode)
{
  int hwEtSum=0;
  for(int etaNr=localEtaMin;etaNr<=localEtaMax;etaNr++){
    for(int phiNr=localPhiMin;phiNr<=localPhiMax;phiNr++){
      
      int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta,etaNr);
      int towerIPhi = l1t::CaloStage2Nav::offsetIPhi(iPhi,phiNr);
      if(abs(towerIEta)<=iEtaAbsMax){
	const l1t::CaloTower& tower = getTower(towers,towerIEta,towerIPhi);
	if(etMode&ECAL) hwEtSum+=tower.hwEtEm();
	if(etMode&HCAL) hwEtSum+=tower.hwEtHad();
      }	
    }
  }
  return hwEtSum;
}


size_t l1t::CaloTools::calNrTowers(int iEtaMin,int iEtaMax,int iPhiMin,int iPhiMax,const std::vector<l1t::CaloTower>& towers,int minHwEt,int maxHwEt,SubDet etMode)
{
  size_t nrTowers=0;
  l1t::CaloStage2Nav nav(iEtaMin,iPhiMin);
  while(nav.currIEta()<=iEtaMax){
    while(nav.currIPhi()<=iPhiMax){
      nav.north();
      const l1t::CaloTower& tower = l1t::CaloTools::getTower(towers,nav.currIEta(),nav.currIPhi());
      int towerHwEt =0;
      if(etMode&ECAL) towerHwEt+=tower.hwEtEm();
      if(etMode&HCAL) towerHwEt+=tower.hwEtHad();
      if(etMode&CALO) towerHwEt+=tower.hwPt();
      if(towerHwEt>=minHwEt && towerHwEt<=maxHwEt) nrTowers++;
    }
    nav.east();
    nav.resetIPhi();
  }
  return nrTowers;
}

std::pair<float,float> l1t::CaloTools::towerEtaBounds(int ieta)
{
  if(ieta==0) ieta = 1;
  if(ieta>32) ieta = 32;
  if(ieta<-32) ieta = -32;
  const float towerEtas[33] = {0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.870,0.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,2.322,2.5,2.650,3.000,3.5,4.0,4.5,5.0}; 
  return std::make_pair( towerEtas[abs(ieta)-1],towerEtas[abs(ieta)] );
}

float l1t::CaloTools::towerEta(int ieta)
{
  std::pair<float,float> bounds = towerEtaBounds(ieta);
  float eta = (bounds.second+bounds.first)/2.;
  float sign = ieta>0 ? 1. : -1.;
  return sign*eta; 
}

float l1t::CaloTools::towerPhi(int ieta, int iphi)
{
  return (float(iphi)-0.5)*towerPhiSize(ieta);
}

float l1t::CaloTools::towerEtaSize(int ieta)
{
  std::pair<float,float> bounds = towerEtaBounds(ieta);
  float size = (bounds.second-bounds.first);
  return size;
}

float l1t::CaloTools::towerPhiSize(int ieta)
{
  if(abs(ieta)<=28) return 2.*M_PI/72.;
  else return 2.*M_PI/18.;
}


