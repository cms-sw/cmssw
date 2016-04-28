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

#ifndef Stage2Layer2EtSumAlgorithmFirmware_H
#define Stage2Layer2EtSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2EtSumAlgorithmFirmwareImp1 : public Stage2Layer2EtSumAlgorithm {
  public:
    Stage2Layer2EtSumAlgorithmFirmwareImp1(CaloParamsHelper* params);
    virtual ~Stage2Layer2EtSumAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::EtSum> & sums);
  private:
    CaloParamsHelper* params_;
    int32_t metTowThresholdHw_;
    int32_t metTowThresholdHw2_;
    int32_t ettTowThresholdHw_;
    int32_t metEtaMax_;
    int32_t metEtaMax2_;
    int32_t ettEtaMax_;
	int32_t cos_coeff[72] = {1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89, 0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019};

	int32_t sin_coeff[72] = {0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019, 1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89};
  };

}

#endif
