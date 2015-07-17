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
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t
{

  // Imp1 is for v1 and v2
  class Stage2Layer2EGammaAlgorithmFirmwareImp1 : public Stage2Layer2EGammaAlgorithm
  {
    public:
      Stage2Layer2EGammaAlgorithmFirmwareImp1(CaloParamsHelper* params); //const CaloMainProcessorParams & dbPars);
      virtual ~Stage2Layer2EGammaAlgorithmFirmwareImp1();
      virtual void processEvent(const std::vector<CaloCluster>& clusters, const std::vector<CaloTower>& towers, std::vector<EGamma>& egammas);

    private:
      // trimming
      l1t::CaloCluster trimCluster(const l1t::CaloCluster& clus);
      unsigned int trimmingLutIndex(unsigned int shape, int iEta);
      // identification
      bool idHOverE(const l1t::CaloCluster& clus, int hwPt);
      unsigned int idHOverELutIndex(int iEta, int E);
      bool idShape(const l1t::CaloCluster& clus, int hwPt);
      unsigned int idShapeLutIndex(int iEta, int E, int shape);
      // isolation
      int isoCalEgHwFootPrint(const l1t::CaloCluster&,const std::vector<l1t::CaloTower>&);
      unsigned isoLutIndex(int iEta,unsigned int nrTowers);
      // calibration
      int calibratedPt(const l1t::CaloCluster& clus, int hwPt);
      unsigned int calibrationLutIndex(int iEta, int E, int shape);

    private:
      CaloParamsHelper* params_;

  };

}

#endif
