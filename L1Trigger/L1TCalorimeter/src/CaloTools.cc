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
size_t caloTowerHash(int iEta,int iPhi)
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
			      int etaMin,int etaMax,int phiMin,int phiMax,SubDet etMode)
{
  int hwEtSum=0;
  for(int etaNr=etaMin;etaNr<=etaMax;etaNr++){
    for(int phiNr=phiMin;phiNr<=phiMax;phiNr++){
      
      int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta,etaNr);
      int towerIPhi = l1t::CaloStage2Nav::offsetIEta(iPhi,phiNr);
      
      const l1t::CaloTower& tower = getTower(towers,towerIEta,towerIPhi);
      if(etMode&ECAL) hwEtSum+=tower.hwEtEm();
      if(etMode&HCAL) hwEtSum+=tower.hwEtHad();
    }
  }
  return hwEtSum;
}


