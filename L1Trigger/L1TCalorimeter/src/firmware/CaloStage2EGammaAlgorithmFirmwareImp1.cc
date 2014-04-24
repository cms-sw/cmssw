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
  unsigned lutIndex(int iEta,unsigned int nrTowers);//also needs a permenant home
  void genLUT(LUT& lut);//a temporary function to generate the LUT (I couldnt bring myself to have a hardcoded filename)
}

l1t::CaloStage2EGammaAlgorithmFirmwareImp1::CaloStage2EGammaAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params),
  lut_(32,10)
{
  genLUT(lut_);

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
   
    int nrTowers = CaloTools::calNrTowers(-4,4,1,72,towers,1,999,CaloTools::CALO);
    unsigned int lutAddress = lutIndex(egammas.back().hwEta(),nrTowers);
   
    int isolBit = hwEtSum-hwFootPrint <= lut_.data(lutAddress); 
    //std::cout <<"hwEtSum "<<hwEtSum<<" hwFootPrint "<<hwFootPrint<<" isol "<<hwEtSum-hwFootPrint<<" bit "<<isolBit<<std::endl;
    
    egammas.back().setHwIso(isolBit);
  }
}


//calculates the footprint of the electron in hardware values
int l1t::calEgHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers)
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();

  // hwEmEtSumLeft =  CaloTools::calHwEtSum(iEta,iPhi,towers,-1,-1,-1,1,CaloTools::ECAL);
  // int hwEmEtSumRight = CaloTools::calHwEtSum(iEta,iPhi,towers,1,1,-1,1,CaloTools::ECAL);
  
  int etaSide = clus.checkClusterFlag(CaloCluster::TRIM_LEFT) ? 1 : -1; //if we trimed left, its the right (ie +ve) side we want
  int phiSide = iEta>0 ? 1 : -1;

  int ecalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,2,2,CaloTools::ECAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,etaSide,etaSide,2,2,CaloTools::ECAL);
  int hcalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,0,0,CaloTools::HCAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,CaloTools::HCAL);
  return ecalHwFootPrint+hcalHwFootPrint;

}

//ieta =-28, nrTowers 0 is 0, increases to ieta28, nrTowers=kNrTowersInSum
unsigned l1t::lutIndex(int iEta,unsigned int nrTowers)
{
  const unsigned int kNrTowersInSum=72*4*2;
  const unsigned int kTowerGranularity=1;
  const unsigned int kMaxAddress = kNrTowersInSum%kTowerGranularity==0 ? (kNrTowersInSum/kTowerGranularity+1)*28*2 : 
                                                                         (kNrTowersInSum/kTowerGranularity)*28*2;
  
  unsigned int nrTowersNormed = nrTowers/kTowerGranularity;
  
  unsigned int iEtaNormed = iEta+28;
  if(iEta>0) iEtaNormed--; //we skip zero
  
  if(std::abs(iEta)>28 || iEta==0 || nrTowers>kNrTowersInSum) return kMaxAddress;
  else return iEtaNormed*(kNrTowersInSum/kTowerGranularity+1)+nrTowersNormed;
  
}

#include <sstream>
void l1t::genLUT(LUT& lut)
{
  std::stringstream stream;
  int address=0;
  for(int iEta=-28;iEta<=28;iEta++){
    if(iEta==0) continue;
    for(int nrTowers=0;nrTowers<=72*4*2;nrTowers++){
      stream <<address<<" "<<nrTowers/4<<std::endl;
      address++;
    }
  }
  lut.read(stream);  
  //std::cout <<"writing lut "<<std::endl;
  //lut.write(std::cout);
}
