#include "DataFormats/Common/interface/CommonExceptions.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
namespace edm {
  void
  checkForWrongProduct(ProductID const& keyID, ProductID const& refID) {
    if (keyID != refID) {
      throw Exception(errors::InvalidReference) <<
	 "AssociationVector: trying to use [] operator passing a reference\n" <<
	 " with the wrong product id (i.e.: pointing to the wrong collection)\n" <<
         " keyRef.id = " << keyID << ", ref.id = " << refID << "\n";
      
    }
  }
}

