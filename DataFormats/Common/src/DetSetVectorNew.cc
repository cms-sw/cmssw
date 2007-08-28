#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmNew {
  namespace dstvdetails {
    void errorFilling() {
      throw edm::Exception(edm::errors::LogicError,"Instantiating a second DetSetVector::FastFiller")
	<< "only one DetSetVector::FastFiller can be active at a given time!";
    }
    void errorIdExists(det_id_type iid) {
      throw edm::Exception(edm::errors::InvalidReference)
	<< "DetSetVector::inserv called with index already in collection;\n"
	<< "index value: " << iid;
    }

    void throw_range(det_id_type iid) {
      throw edm::Exception(edm::errors::InvalidReference)
	<< "DetSetVector::operator[] called with index not in collection;\n"
	<< "index value: " << iid;
    }
 
  }
}
