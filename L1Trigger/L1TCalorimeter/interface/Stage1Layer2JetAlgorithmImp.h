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

#ifndef L1TCALOSTAGE1JETALGORITHMIMP_H
#define L1TCALOSTAGE1JETALGORITHMIMP_H

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2JetAlgorithm.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  class Stage1Layer2JetAlgorithmImpHI : public Stage1Layer2JetAlgorithm {
  public:
    Stage1Layer2JetAlgorithmImpHI(CaloParams* params);
    virtual ~Stage1Layer2JetAlgorithmImpHI();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      std::vector<l1t::Jet> * jets);
  private:
    CaloParams* const params_;
    //double regionLSB_;
  };

  class Stage1Layer2JetAlgorithmImpPP : public Stage1Layer2JetAlgorithm {
  public:
    Stage1Layer2JetAlgorithmImpPP(CaloParams* params);
    virtual ~Stage1Layer2JetAlgorithmImpPP();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      std::vector<l1t::Jet> * jets);
  private:
    CaloParams* const params_;
    //double regionLSB_;
  };
}

#endif
