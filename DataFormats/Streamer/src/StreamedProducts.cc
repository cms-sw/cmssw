#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"

namespace edm {
    StreamedProduct::StreamedProduct(EDProduct const* prod,
		    BranchDescription const& desc,
		    ModuleDescriptionID const& mod,
		    ProductID pid,
		    ProductStatus status,
		    std::vector<BranchID> const* parents) :
      prod_(prod), desc_(&desc), mod_(mod), productID_(pid), status_(status), parents_(parents) {
      if (productstatus::present(status_) && (prod == 0 || !prod->isPresent())) {
	desc.init();
        throw edm::Exception(edm::errors::LogicError, "StreamedProduct::StreamedProduct\n")
           << "A product with a status of 'present' is not actually present.\n"
           << "The branch name is " << desc.branchName() << "\n"
           << "Contact a framework developer.\n";
      }
    }
}

