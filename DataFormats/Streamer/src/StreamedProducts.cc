#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
    StreamedProduct::StreamedProduct(EDProduct const* prod,
		    BranchDescription const& desc,
		    ProductStatus status,
		    std::vector<BranchID> const* parents) :
      prod_(prod), desc_(&desc), status_(status), parents_(parents) {
      if (productstatus::present(status_) && (prod == 0 || !prod->isPresent())) {
	desc.init();
        throw edm::Exception(edm::errors::LogicError, "StreamedProduct::StreamedProduct\n")
           << "A product with a status of 'present' is not actually present.\n"
           << "The branch name is " << desc.branchName() << "\n"
           << "Contact a framework developer.\n";
      }
    }
}

