///
/// \class l1t::CaloStage2JetAlgorithm
///
///

//

#ifndef Stage1Layer2JetAlgorithm_h
#define Stage1Layer2JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include <vector>

namespace l1t {

  class Stage1Layer2JetAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      std::vector<l1t::Jet> * jets,
			      std::vector<l1t::Jet> * preGtJets) = 0;
    virtual ~Stage1Layer2JetAlgorithm(){};

    /* private: */
    /*   double regionLSB_; // least significant bit value. Usually = 0.5 */
  };

}

#endif
