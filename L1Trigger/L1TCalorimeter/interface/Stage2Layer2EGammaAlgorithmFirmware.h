///
/// Description: Firmware headers
///
/// Implementation:
///    Concrete firmware implementations
///
/// \author: Jim Brooke - University of Bristol
///

//
//

#ifndef Stage2Layer2EGammaAlgorithmFirmware_H
#define Stage2Layer2EGammaAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EGammaAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2EGammaAlgorithmFirmwareImp1 : public Stage2Layer2EGammaAlgorithm {
  public:
    Stage2Layer2EGammaAlgorithmFirmwareImp1(CaloParams* params); //const CaloMainProcessorParams & dbPars);
    virtual ~Stage2Layer2EGammaAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters, 
			      const std::vector<CaloTower>& towers,
			      std::vector<EGamma> & egammas);
    
  private:
    int calEgHwFootPrint(const l1t::CaloCluster&,const std::vector<l1t::CaloTower>&);
    unsigned lutIndex(int iEta,unsigned int nrTowers);
    
  private:
    CaloParams* params_;

  };
  
}

#endif
