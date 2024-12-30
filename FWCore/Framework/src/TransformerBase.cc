#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <optional>

namespace {
  class TransformSignalSentry {
  public:
    TransformSignalSentry(edm::ActivityRegistry* a, edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
        : a_(a), sc_(sc), mcc_(mcc) {
      if (a_)
        a_->preModuleTransformSignal_(sc_, mcc_);
    }
    ~TransformSignalSentry() {
      if (a_)
        a_->postModuleTransformSignal_(sc_, mcc_);
    }

  private:
    edm::ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
    edm::StreamContext const& sc_;
    edm::ModuleCallingContext const& mcc_;
  };

  class TransformAcquiringSignalSentry {
  public:
    TransformAcquiringSignalSentry(edm::ActivityRegistry* a,
                                   edm::StreamContext const& sc,
                                   edm::ModuleCallingContext const& mcc)
        : a_(a), sc_(sc), mcc_(mcc) {
      if (a_)
        a_->preModuleTransformAcquiringSignal_(sc_, mcc_);
    }
    ~TransformAcquiringSignalSentry() {
      if (a_)
        a_->postModuleTransformAcquiringSignal_(sc_, mcc_);
    }

  private:
    edm::ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
    edm::StreamContext const& sc_;
    edm::ModuleCallingContext const& mcc_;
  };

}  // namespace

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
                                                 edm::BranchDescription const& iBranch) const noexcept {
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
                                          edm::ActivityRegistry* iAct,
                                          ProducerBase const& iBase,
                                          edm::EventForTransformer& iEvent) const noexcept {
    auto const& mcc = iEvent.moduleCallingContext();
    if (transformInfo_.get<kPreTransform>(iIndex)) {
      std::optional<decltype(iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex)))>
          handle;
      //transform acquiring signal
      auto const& streamContext = *mcc.getStreamContext();
      TransformAcquiringSignalSentry sentry(iAct, streamContext, mcc);
      CMS_SA_ALLOW try {
        handle = iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex));
      } catch (...) {
        iHolder.doneWaiting(std::current_exception());
        return;
      }
      if (handle->wrapper()) {
        auto cache = std::make_shared<std::any>();
        auto nextTask =
            edm::make_waiting_task([holder = iHolder, cache, iIndex, this, &iBase, handle = *handle, iEvent, iAct](
                                       std::exception_ptr const* iPtr) mutable {
              if (iPtr) {
                holder.doneWaiting(*iPtr);
              } else {
                //transform signal
                auto mcc = iEvent.moduleCallingContext();
                auto const& streamContext = *mcc.getStreamContext();
                TransformSignalSentry sentry(iAct, streamContext, mcc);
                iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                           transformInfo_.get<kTransform>(iIndex)(streamContext.streamID(), std::move(*cache)),
                           handle);
              }
            });
        WaitingTaskWithArenaHolder wta(*iHolder.group(), nextTask);
        CMS_SA_ALLOW try {
          *cache = transformInfo_.get<kPreTransform>(iIndex)(streamContext.streamID(), *(handle->wrapper()), wta);
        } catch (...) {
          wta.doneWaiting(std::current_exception());
        }
      }
    } else {
      CMS_SA_ALLOW try {
        auto handle = iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex));

        if (handle.wrapper()) {
          std::any v = handle.wrapper();
          //transform signal
          auto const& streamContext = *mcc.getStreamContext();
          TransformSignalSentry sentry(iAct, streamContext, mcc);
          iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                     transformInfo_.get<kTransform>(iIndex)(streamContext.streamID(), std::move(v)),
                     handle);
        }
      } catch (...) {
        iHolder.doneWaiting(std::current_exception());
      }
    }
  }

}  // namespace edm
