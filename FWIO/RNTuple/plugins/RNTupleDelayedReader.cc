#include "RNTupleDelayedReader.h"
#include "DataProductsRNTuple.h"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"

#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "FWCore/Utilities/interface/EDMException.h"

namespace edm::input {
  RNTupleDelayedReader::RNTupleDelayedReader(DataProductsRNTuple* iRNTuple,
                                             SharedResourcesAcquirer* iAcquirer,
                                             std::recursive_mutex* iMutex)
      : rntuple_(iRNTuple), resourceAcquirer_(iAcquirer), mutex_(iMutex) {}

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> RNTupleDelayedReader::sharedResources_() const {
    return std::make_pair(resourceAcquirer_, mutex_);
  }

  std::shared_ptr<WrapperBase> RNTupleDelayedReader::getProduct_(BranchID const& k, EDProductGetter const* ep) {
    if (lastException_) {
      try {
        std::rethrow_exception(lastException_);
      } catch (edm::Exception const& e) {
        //avoid growing the context each time the exception is rethrown.
        auto copy = e;
        copy.addContext("Rethrowing an exception that happened on a different read request.");
        throw copy;
      } catch (cms::Exception& e) {
        //If we do anything here to 'copy', we would lose the actual type of the exception.
        e.addContext("Rethrowing an exception that happened on a different read request.");
        throw;
      }
    }

    setRefCoreStreamer(ep);
    //make code exception safe
    std::shared_ptr<void> refCoreStreamerGuard(nullptr, [](void*) { setRefCoreStreamer(false); });

    std::shared_ptr<WrapperBase> edp;
    try {
      edp = rntuple_->dataProduct(k, entry_);
    } catch (...) {
      lastException_ = std::current_exception();
      std::rethrow_exception(lastException_);
    }
    //if (rntuple_->branchType() == InEvent) {
    // CMS-THREADING For the primary input source calls to this function need to be serialized
    //InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    //}

    return edp;
  }
}  // namespace edm::input
