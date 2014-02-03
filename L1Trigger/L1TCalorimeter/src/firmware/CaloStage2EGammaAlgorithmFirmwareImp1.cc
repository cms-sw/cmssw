///
/// \class l1t::CaloStage2EGammaAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"


//NOTE: this is NOT finished and doesnt do anything. 

//these functions will be implimented and put somewhere else in other class asap, they are only here as placeholders
int calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>&){return 0;}
int calEgHwFootPrint(const l1t::EGamma&,const std::vector<l1t::CaloTower>&){return 0;}
int caloTowerHash(int iEta,int iPhi);
const l1t::CaloTower& getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi);

l1t::CaloStage2EGammaAlgorithmFirmwareImp1::CaloStage2EGammaAlgorithmFirmwareImp1() {


}


l1t::CaloStage2EGammaAlgorithmFirmwareImp1::~CaloStage2EGammaAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2EGammaAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      const std::vector<l1t::CaloTower>& towers,
							      std::vector<l1t::EGamma> & egammas) {
  egammas.clear();
  for(size_t clusNr=0;clusNr<clusters.size();clusNr++){
    egammas.push_back(clusters[clusNr]);

    int hwEtSum = calHwEtSum(clusters[clusNr].hwEta(),clusters[clusNr].hwPhi(),towers);
    int hwFootPrint = calEgHwFootPrint(clusters[clusNr],towers);
    egammas.back().setHwIso(hwEtSum-hwFootPrint);
  }
  


}

int calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,int etaMin,int etaMax,int phiMin,int phiMax,int etMode)
{
  int hwEtSum=0;
  for(int etaNr=etaMin;etaNr<=etaMax;etaNr++){
    for(int phiNr=phiMin;phiNr<=phiMax;phiNr++){
      
      int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta,etaNr);
      int towerIPhi = l1t::CaloStage2Nav::offsetIEta(iPhi,phiNr);
      
      const l1t::CaloTower& tower = getTower(towers,towerIEta,towerIPhi);
      if(etMode&0x1) hwEtSum+=tower.hwEtEm();
      if(etMode&0x2) hwEtSum+=tower.hwEtHad();
    }
  }
  return hwEtSum;
}

//calculates the footprint of the electron in hardware values
int calEgHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers)
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();

  int hwEmEtSumLeft = calHwEtSum(iEta,iPhi,towers,-1,-1,-1,1,1);
  int hwEmEtSumRight = calHwEtSum(iEta,iPhi,towers,1,1,-1,1,1);
  
  int etaSide = hwEmEtSumLeft>hwEmEtSumRight ? 1 : -1;
  int phiSide = iEta>0 ? 1 : -1;

  int ecalHwFootPrint = calHwEtSum(iEta,iPhi,towers,0,0,2,2,1)+calHwEtSum(iEta,iPhi,towers,etaSide,etaSide,2,2,1);
  int hcalHwFootPrint = calHwEtSum(iEta,iPhi,towers,0,0,0,0,2)+calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,2);
  return ecalHwFootPrint+hcalHwFootPrint;

}

const l1t::CaloTower& getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi)
{
  static l1t::CaloTower nullTower; //we return this if no tower is found
  for(size_t towerNr=0;towerNr<towers.size();towerNr++){
    if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
  }
  return nullTower;
}

//starts from zero, idea is to make a vector index
int caloTowerHash(int iEta,int iPhi)
{
  return (iEta-1)*72+iPhi-1;
}
