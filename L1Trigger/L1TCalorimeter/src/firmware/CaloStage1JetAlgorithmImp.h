///
/// Description: Firmware headers
///
/// Implementation:
/// Collects concrete firmware implmentations.
///
/// \author: R. Alex Barbieri MIT
///

//
// This header file contains the class definitions for all of the concrete
// implementations of the firmware interface. The CaloStage1JetAlgorithmFactory
// selects the appropriate implementation based on the firmware version in the
// configuration.
//

#ifndef L1TCALOSTAGE1JETALGORITHMIMP_H
#define L1TCALOSTAGE1JETALGORITHMIMP_H

#include "L1Trigger/L1TYellow/interface/CaloStage1JetAlgorithm.h"
#include "L1Trigger/L1TYellow/interface/CaloStage1JetAlgorithmFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage1JetAlgorithmImpHI : public CaloStage1JetAlgorithm {
  public:
    CaloStage1JetAlgorithmImpHI(const CaloParams & dbPars);
    virtual ~CaloStage1JetAlgorithmImpHI();
    virtual void processEvent(const BXVector<l1t::CaloRegion> & regions,
			      BXVector<l1t::Jet> & jets);
  private:
    CaloParams const & db;
  };

  /* // Imp2 is for v3 */
  /* class CaloStage1JetAlgorithmImp2 : public CaloStage1JetAlgorithm { */
  /* public: */
  /*   CaloStage1JetAlgorithmImp2(const CaloParams & dbPars); */
  /*   virtual ~CaloStage1JetAlgorithmImp2(); */
  /*   virtual void processEvent(const BXVector<l1t::CaloRegion> & regions,
			      BXVector<l1t::Jet> & jets); */
  /* private: */
  /*   CaloParams const & db; */
  /* }; */

}

#endif
