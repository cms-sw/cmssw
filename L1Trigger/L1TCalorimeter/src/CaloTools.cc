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
    if(towers[towerIndex].hwEta()!=iEta || towers[towerIndex].hwPhi()!=iPhi){ 
      for(size_t towerNr=0;towerNr<towers.size();towerNr++){
	if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
      }     
    }else return towers[towerIndex];  
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
