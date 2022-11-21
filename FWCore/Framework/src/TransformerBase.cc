#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <optional>

namespace edm {
  void TransformerBase::registerTransformImp(
      ProducerBase& iBase, EDPutToken iToken, const TypeID& id, std::string instanceName, TransformFunction iFunc) {
    auto transformPut = iBase.transforms(id, std::move(instanceName));
    PreTransformFunction ptf;
    transformInfo_.emplace_back(iToken.index(), id, transformPut, std::move(ptf), std::move(iFunc));
  }

  void TransformerBase::registerTransformAsyncImp(ProducerBase& iBase,
                                                  EDPutToken iToken,
                                                  const TypeID& id,
                                                  std::string instanceName,
                                                  PreTransformFunction iPreFunc,
                                                  TransformFunction iFunc) {
    auto transformPut = iBase.transforms(id, std::move(instanceName));
    transformInfo_.emplace_back(iToken.index(), id, transformPut, std::move(iPreFunc), std::move(iFunc));
  }

  std::size_t TransformerBase::findMatchingIndex(ProducerBase const& iBase,
                                                 edm::BranchDescription const& iBranch) const {
    auto const& list = iBase.typeLabelList();

    std::size_t index = 0;
    bool found = false;
    for (auto const& element : list) {
      if (not element.isTransform_) {
        continue;
      }
      if (element.typeID_ == iBranch.unwrappedTypeID() &&
          element.productInstanceName_ == iBranch.productInstanceName()) {
        found = true;
        break;
      }
      ++index;
    }
    assert(found);
    return index;
  }

  void TransformerBase::extendUpdateLookup(ProducerBase const& iBase,
                                           ModuleDescription const& iModuleDesc,
                                           ProductResolverIndexHelper const& iHelper) {
    auto const& list = iBase.typeLabelList();

    for (auto it = transformInfo_.begin<0>(); it != transformInfo_.end<0>(); ++it) {
      auto const& putInfo = list[*it];
      *it = iHelper.index(PRODUCT_TYPE,
                          putInfo.typeID_,
                          iModuleDesc.moduleLabel().c_str(),
                          putInfo.productInstanceName_.c_str(),
                          iModuleDesc.processName().c_str());
    }
  }

  void TransformerBase::transformImpAsync(edm::WaitingTaskHolder iHolder,
                                          std::size_t iIndex,
                                          ProducerBase const& iBase,
                                          edm::EventForTransformer& iEvent) const {
    if (transformInfo_.get<kPreTransform>(iIndex)) {
      std::optional<decltype(iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex)))>
          handle;
      CMS_SA_ALLOW try {
        handle = iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex));
      } catch (...) {
        iHolder.doneWaiting(std::current_exception());
        return;
      }
      if (handle->wrapper()) {
        auto cache = std::make_shared<std::any>();
        auto nextTask =
            edm::make_waiting_task([holder = iHolder, cache, iIndex, this, &iBase, handle = *handle, iEvent](
                                       std::exception_ptr const* iPtr) mutable {
              if (iPtr) {
                holder.doneWaiting(*iPtr);
              } else {
                iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                           transformInfo_.get<kTransform>(iIndex)(std::move(*cache)),
                           handle);
              }
            });
        WaitingTaskWithArenaHolder wta(*iHolder.group(), nextTask);
        CMS_SA_ALLOW try {
          *cache = transformInfo_.get<kPreTransform>(iIndex)(*(handle->wrapper()), wta);
        } catch (...) {
          wta.doneWaiting(std::current_exception());
        }
      }
    } else {
      CMS_SA_ALLOW try {
        auto handle = iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex));

        if (handle.wrapper()) {
          std::any v = handle.wrapper();
          iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                     transformInfo_.get<kTransform>(iIndex)(std::move(v)),
                     handle);
        }
      } catch (...) {
        iHolder.doneWaiting(std::current_exception());
      }
    }
  }

}  // namespace edm
