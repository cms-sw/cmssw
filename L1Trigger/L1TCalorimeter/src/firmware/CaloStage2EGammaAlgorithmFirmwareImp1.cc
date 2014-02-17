///
/// \class l1t::CaloStage2EGammaAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


//NOTE: this is NOT finished and doesnt do anything. 

namespace l1t{
  int calEgHwFootPrint(const l1t::CaloCluster&,const std::vector<l1t::CaloTower>&);//still needs a permenant home
}

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

    int hwEtSum = CaloTools::calHwEtSum(clusters[clusNr].hwEta(),clusters[clusNr].hwPhi(),towers,-2,2,-3,3);
    int hwFootPrint = calEgHwFootPrint(clusters[clusNr],towers);
    egammas.back().setHwIso(hwEtSum-hwFootPrint);
  }
}


//calculates the footprint of the electron in hardware values
int l1t::calEgHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers)
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();

  int hwEmEtSumLeft =  CaloTools::calHwEtSum(iEta,iPhi,towers,-1,-1,-1,1,CaloTools::ECAL);
  int hwEmEtSumRight = CaloTools::calHwEtSum(iEta,iPhi,towers,1,1,-1,1,CaloTools::ECAL);
  
  int etaSide = hwEmEtSumLeft>hwEmEtSumRight ? 1 : -1;
  int phiSide = iEta>0 ? 1 : -1;

  int ecalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,2,2,CaloTools::ECAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,etaSide,etaSide,2,2,CaloTools::ECAL);
  int hcalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,0,0,CaloTools::HCAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,CaloTools::HCAL);
  return ecalHwFootPrint+hcalHwFootPrint;

}
