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

#ifndef Stage2Layer2JetSumAlgorithmFirmware_H
#define Stage2Layer2JetSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2JetSumAlgorithmFirmwareImp1 : public Stage2Layer2JetSumAlgorithm {
  public:
    Stage2Layer2JetSumAlgorithmFirmwareImp1(CaloParamsHelper* params);
    virtual ~Stage2Layer2JetSumAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::Jet> & alljets,
			      std::vector<l1t::EtSum> & htsums);
  private:
    CaloParamsHelper* params_;
    int32_t mhtJetThresholdHw_;
    int32_t httJetThresholdHw_;
    int32_t mhtEtaMax_;
    int32_t httEtaMax_;
	int32_t cos_coeff[72] = {1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89, 0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019};

	int32_t sin_coeff[72] = {0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019, 1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89};
  };

}

#endif
