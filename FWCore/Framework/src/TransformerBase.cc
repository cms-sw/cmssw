#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

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
                                                 edm::ProductDescription const& iBranch) const noexcept {
    auto const& list = iBase.typeLabelList();

    std::size_t index = 0;
    [[maybe_unused]] bool found = false;
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
      auto sentry = signalslot::make_sentry([iAct, &streamContext, &mcc]() {
        if (iAct) {
          iAct->postModuleTransformAcquiringSignal_.emit(streamContext, mcc);
        }
      });
      if (iAct) {
        iAct->preModuleTransformAcquiringSignal_.emit(streamContext, mcc);
      }
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
                auto sentry = signalslot::make_sentry([iAct, &streamContext, &mcc]() {
                  if (iAct) {
                    iAct->postModuleTransformSignal_.emit(streamContext, mcc);
                  }
                });
                if (iAct) {
                  iAct->preModuleTransformSignal_.emit(streamContext, mcc);
                }
                iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                           transformInfo_.get<kTransform>(iIndex)(streamContext.streamID(), std::move(*cache)),
                           handle);
                sentry.succeeded();
              }
            });
        WaitingTaskHolder wth(*iHolder.group(), nextTask);
        CMS_SA_ALLOW try {
          // wth must be copied into wta below so that the
          // wth.doneWaiting() is called after the pre-transform
          // function has finished
          WaitingTaskWithArenaHolder wta(wth);
          *cache =
              transformInfo_.get<kPreTransform>(iIndex)(streamContext.streamID(), *(handle->wrapper()), std::move(wta));
        } catch (...) {
          wth.doneWaiting(std::current_exception());
        }
      }
      sentry.succeeded();
    } else {
      CMS_SA_ALLOW try {
        auto handle = iEvent.get(transformInfo_.get<kType>(iIndex), transformInfo_.get<kResolverIndex>(iIndex));

        if (handle.wrapper()) {
          std::any v = handle.wrapper();
          //transform signal
          auto const& streamContext = *mcc.getStreamContext();
          auto sentry = signalslot::make_sentry([iAct, &streamContext, &mcc]() {
            if (iAct) {
              iAct->postModuleTransformSignal_.emit(streamContext, mcc);
            }
          });
          if (iAct) {
            iAct->preModuleTransformSignal_.emit(streamContext, mcc);
          }
          iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<kToken>(iIndex).index()],
                     transformInfo_.get<kTransform>(iIndex)(streamContext.streamID(), std::move(v)),
                     handle);
          sentry.succeeded();
        }
      } catch (...) {
        iHolder.doneWaiting(std::current_exception());
      }
    }
  }

}  // namespace edm
