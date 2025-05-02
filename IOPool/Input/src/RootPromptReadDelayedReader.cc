/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootPromptReadDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"

#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "TBranch.h"
#include "TClass.h"
#include "TTree.h"

#include <cassert>

namespace edm {

  RootPromptReadDelayedReader::RootPromptReadDelayedReader(RootTree const& tree,
                                                           std::shared_ptr<InputFile> filePtr,
                                                           InputType inputType,
                                                          unsigned int iNIndexes)
      : cacheMaps_(iNIndexes), tree_(tree), filePtr_(filePtr), nextReader_(), inputType_(inputType) {
    if (inputType == InputType::Primary) {
      auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
      resourceAcquirer_ = std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
      mutex_ = resources.second;
    }
  }

  RootPromptReadDelayedReader::~RootPromptReadDelayedReader() {}

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> RootPromptReadDelayedReader::sharedResources_() const {
    return std::make_pair(resourceAcquirer_.get(), mutex_.get());
  }

  namespace {
    unsigned int indexFor(RootTree const& tree, EDProductGetter const* ep) {
      return tree.branchType() == InEvent ? ep->transitionIndex() : 0;
    }
  }
  std::shared_ptr<WrapperBase> RootPromptReadDelayedReader::getProduct_(BranchID const& k, EDProductGetter const* ep) {
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
    auto& cacheMap = cacheMaps_[indexFor(tree_, ep)];
    auto itFound = cacheMap.find(k.id());
    if (itFound != cacheMap.end()) {
      auto& cache = itFound->second;
      if (cache.wrapperBase_) {
        if (tree_.branchType() == InEvent) {
          // CMS-THREADING For the primary input source calls to this function need to be serialized
          InputFile::reportReadBranch(inputType_, std::string(tree_.branches().find(itFound->first)->productBranch_->GetName()));
        }
        return std::shared_ptr<WrapperBase>(std::move(cache.wrapperBase_));
      }
    } 
    if (nextReader_) {
      return nextReader_->getProduct(k, ep);
    }
    return std::shared_ptr<WrapperBase>();
  }

  void RootPromptReadDelayedReader::readAllProductsNow(EDProductGetter const* ep) {
    // first set all the addresses
    auto& cacheMap = cacheMaps_[indexFor(tree_, ep)];
    if (cacheMap.empty()) {
      for(auto& cacheMap : cacheMaps_) {
        cacheMap.reserve(tree_.branches().size());
        for(auto const& branch : tree_.branches()) {
          cacheMap.emplace(branch.first, Cache{});
        }
      }
    }
    for (auto& it : cacheMap) {
      auto branchInfo = getBranchInfo(it.first);
      if (branchInfo == nullptr || branchInfo->productBranch_ == nullptr) {
        continue;  // Skip if branch info or product branch is not available
      }
      auto& cache = it.second;
      cache.wrapperBase_ = branchInfo->newWrapper();
      cache.wrapperBasePtr_ = cache.wrapperBase_.get();
      branchInfo->productBranch_->SetAddress(&cache.wrapperBasePtr_);
    }

    setRefCoreStreamer(ep);
    //make code exception safe
    std::shared_ptr<void> refCoreStreamerGuard(nullptr, [](void*) {
      setRefCoreStreamer(false);
      ;
    });

    tree_.getEntryForAllBranches();
  }
}  // namespace edm
