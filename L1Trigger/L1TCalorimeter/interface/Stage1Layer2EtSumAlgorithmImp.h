///
/// Description: Firmware headers
///
/// Implementation:
/// Collects concrete algorithm implmentations.
///
/// \author: R. Alex Barbieri MIT
///

//
// This header file contains the class definitions for all of the concrete
// implementations of the firmware interface. The Stage1Layer2FirmwareFactory
// selects the appropriate implementation based on the firmware version in the
// configuration.
//

#ifndef L1TCALOSTAGE1ETSUMSALGORITHMIMP_H
#define L1TCALOSTAGE1ETSUMSALGORITHMIMP_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EtSumAlgorithm.h"
//#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsStage1.h"


namespace l1t {

  class Stage1Layer2EtSumAlgorithmImpPP : public Stage1Layer2EtSumAlgorithm {
  public:
    Stage1Layer2EtSumAlgorithmImpPP(CaloParamsStage1* params);
    virtual ~Stage1Layer2EtSumAlgorithmImpPP();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      std::vector<l1t::EtSum> * sums);

    int etSumEtaMinHt;
    int etSumEtaMaxHt;
    double etSumEtThresholdHt;
  
    int etSumEtaMinEt;
    int etSumEtaMaxEt;
    double etSumEtThresholdEt;

    double jetLsb;
  private:
    CaloParamsStage1* const params_;
    double regionPhysicalEt(const l1t::CaloRegion&) const;

    std::vector<double> sinPhi;
    std::vector<double> cosPhi;

  };

  /* class Stage1Layer2CentralityAlgorithm : public Stage1Layer2EtSumAlgorithm { */
  /* public: */
  /*   Stage1Layer2CentralityAlgorithm(CaloParamsStage1* params); */
  /*   virtual ~Stage1Layer2CentralityAlgorithm(); */
  /*   virtual void processEvent(const std::vector<l1t::CaloRegion> & regions, */
  /* 			      const std::vector<l1t::CaloEmCand> & EMCands, */
  /* 			      std::vector<l1t::EtSum> * sums); */
  /* private: */
  /*   CaloParamsStage1* const params_; */
  /* }; */
}

#endif
