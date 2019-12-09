#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/WrapperBase.h"

namespace edm {
  StreamedProduct::StreamedProduct(WrapperBase const* prod,
                                   BranchDescription const& desc,
                                   bool present,
                                   std::vector<BranchID> const* parents)
      : prod_(prod), desc_(&desc), present_(present), parents_(parents) {
    if (present_ && prod == nullptr) {
      std::string branchName = desc.branchName();
      if (branchName.empty()) {
        BranchDescription localCopy(desc);
        localCopy.initBranchName();
        branchName = localCopy.branchName();
      }
      throw edm::Exception(edm::errors::LogicError, "StreamedProduct::StreamedProduct\n")
          << "A product with a status of 'present' is not actually present.\n"
          << "The branch name is " << branchName << "\n"
          << "Contact a framework developer.\n";
    }
  }

  void SendJobHeader::initializeTransients() {
    for (BranchDescription& desc : descs_) {
      desc.init();
      desc.setIsProvenanceSetOnRead();
    }
  }
}  // namespace edm
