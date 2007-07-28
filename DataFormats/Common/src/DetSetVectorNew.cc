#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmNew {
  namespace dstvdetails {
    void errorFilling() {
      throw edm::Exception(edm::errors::LogicError,"Instantiating a second DetSetVector::FastFiller")
	<< "only one DetSetVector::FastFiller can be active at a given time!"
    }
  }
}
