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


