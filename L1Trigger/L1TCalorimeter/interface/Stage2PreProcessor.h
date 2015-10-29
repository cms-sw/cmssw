///
/// \class l1t::Stage2PreProcessor
///
/// Description: interface for the pre-processor
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2PreProcessor_h
#define Stage2PreProcessor_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

namespace l1t {

  class Stage2PreProcessor {
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers) = 0;

    virtual ~Stage2PreProcessor(){};

  };

}

#endif
