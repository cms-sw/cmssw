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
// implementations of the firmware interface. The CaloStage1FirmwareFactory
// selects the appropriate implementation based on the firmware version in the
// configuration.
//

#ifndef L1TCALOSTAGE1JETALGORITHMIMP_H
#define L1TCALOSTAGE1JETALGORITHMIMP_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1JetAlgorithm.h"
//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  class CaloStage1JetAlgorithmImpHI : public CaloStage1JetAlgorithm {
  public:
    CaloStage1JetAlgorithmImpHI(/*const CaloParams & dbPars*/);
    virtual ~CaloStage1JetAlgorithmImpHI();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::Jet> & jets);
  private:
    /*CaloParams const & db;*/
    double regionLSB_;
  };

  class CaloStage1JetAlgorithmImpPP : public CaloStage1JetAlgorithm {
  public:
    CaloStage1JetAlgorithmImpPP(/*const CaloParams & dbPars*/);
    virtual ~CaloStage1JetAlgorithmImpPP();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::Jet> & jets);
  private:
    /*CaloParams const & db;*/
    double regionLSB_;
  };
}

#endif
