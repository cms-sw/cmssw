#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

const l1t::CaloTower l1t::CaloTools::nullTower_;

//currently implimented as a brute force search but this will hopefully change in the future
//with standarising the layout of std::vector<l1t::CaloTower>
const l1t::CaloTower& l1t::CaloTools::getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi)
{
  for(size_t towerNr=0;towerNr<towers.size();towerNr++){
    if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
  }
  return nullTower_;
}

//this implimentation has not all the necessary info yet, we need to check the exact HF numbering
//(iEta=-28,iPhi=1)=index 0 to (iEta=28,iPhi=72)=index 28*72*2-1
//HF then runs after that so -32,1 = 28*72*2
size_t l1t::CaloTools::caloTowerHash(int iEta,int iPhi)
{
  //these constants will be either moved to be class members or read in from a database once a decision on this has been made
  const int kHBHEEnd=28;
  const int kHFBegin=29;
  const int kHFEnd=32;
  const int kHFNrPhi=72/4;
  const int kHBHENrPhi=72;
  const int kNrTowers = ((kHFEnd-kHFBegin)*kHFNrPhi + kHBHEEnd*kHBHENrPhi )*2;
  const int kNrHBHETowers = kHBHEEnd*kHBHENrPhi;
  
  const int absIEta = abs(iEta);

  if(absIEta>kHFEnd) return kNrTowers;
  else if(absIEta<=kHBHEEnd){ //HBHE
    return (iEta+kHBHEEnd)*kHBHENrPhi+iPhi-1;
  }else{ //HF
    int iEtaIndex = iEta+kHFEnd; //iEta=-32 is 0
    if(iEta>0) iEta-=kHBHEEnd*2; //but iEta=29 is 5
    return iEtaIndex*kHFNrPhi+iPhi-1 + kNrHBHETowers;
  }
}

int l1t::CaloTools::calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			      int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,SubDet etMode)
{
  int hwEtSum=0;
  for(int etaNr=localEtaMin;etaNr<=localEtaMax;etaNr++){
    for(int phiNr=localPhiMin;phiNr<=localPhiMax;phiNr++){
      
      int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta,etaNr);
      int towerIPhi = l1t::CaloStage2Nav::offsetIEta(iPhi,phiNr);
      
      const l1t::CaloTower& tower = getTower(towers,towerIEta,towerIPhi);
      if(etMode&ECAL) hwEtSum+=tower.hwEtEm();
      if(etMode&HCAL) hwEtSum+=tower.hwEtHad();
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
      if(towerHwEt>=minHwEt && towerHwEt<=maxHwEt) nrTowers++;
    }
    nav.east();
    nav.resetIPhi();
  }
  return nrTowers;
}
