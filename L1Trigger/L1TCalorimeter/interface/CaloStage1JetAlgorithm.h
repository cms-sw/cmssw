///
/// \class l1t::CaloStage2JetAlgorithm
///
///

//

#ifndef CaloStage1JetAlgorithm_h
#define CaloStage1JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include <vector>

namespace l1t {

  class CaloStage1JetAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::Jet> * jets) = 0;

    virtual ~CaloStage1JetAlgorithm(){};

    /* private: */
    /*   double regionLSB_; // least significant bit value. Usually = 0.5 */
  };

}

#endif
