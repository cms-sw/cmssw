/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "ProductResolvers.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/UnscheduledAuxiliary.h"
#include "UnscheduledConfigurator.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/src/ProductDeletedException.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"
#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <cassert>
#include <utility>

namespace edm {

  void DataManagingProductResolver::throwProductDeletedException() const {
    ProductDeletedException exception;
    exception << "DataManagingProductResolver::resolveProduct_: The product matching all criteria was already deleted\n"
              << "Looking for type: " << productDescription().unwrappedTypeID() << "\n"
              << "Looking for module label: " << moduleLabel() << "\n"
              << "Looking for productInstanceName: " << productInstanceName() << "\n"
              << (processName().empty() ? "" : "Looking for process: ") << processName() << "\n"
              << "This means there is a configuration error.\n"
              << "The module which is asking for this data must be configured to state that it will read this data.";
    throw exception;
  }

  //This is a templated function in order to avoid calling another virtual function
  template <bool callResolver, typename FUNC>
  ProductResolverBase::Resolution DataManagingProductResolver::resolveProductImpl(FUNC resolver) const {
    if (productWasDeleted()) {
      throwProductDeletedException();
    }
    auto presentStatus = status();

    if (callResolver && presentStatus == ProductStatus::ResolveNotRun) {
      //if resolver fails because of exception or not setting product
      // make sure the status goes to failed
      auto failedStatusSetter = [this](ProductStatus* iPresentStatus) {
        if (this->status() == ProductStatus::ResolveNotRun) {
          this->setFailedStatus();
        }
        *iPresentStatus = this->status();
      };
      std::unique_ptr<ProductStatus, decltype(failedStatusSetter)> failedStatusGuard(&presentStatus,
                                                                                     failedStatusSetter);

      //If successful, this will call setProduct
      resolver();
    }

    if (presentStatus == ProductStatus::ProductSet) {
      auto pd = &getProductData();
      if (pd->wrapper()->isPresent()) {
        return Resolution(pd);
      }
    }

    return Resolution(nullptr);
  }

  void MergeableInputProductResolver::mergeProduct(
      std::shared_ptr<WrapperBase> iFrom, MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
    // if its not mergeable and the previous read failed, go ahead and use this one
    if (status() == ProductStatus::ResolveFailed) {
      setProduct(std::move(iFrom));
      return;
    }

    assert(status() == ProductStatus::ProductSet);
    if (not iFrom) {
      return;
    }

    checkType(*iFrom);

    auto original = getProductData().unsafe_wrapper();
    if (original->isMergeable()) {
      if (original->isPresent() != iFrom->isPresent()) {
        throw Exception(errors::MismatchedInputFiles)
            << "Merge of Run or Lumi product failed for branch " << productDescription().branchName() << "\n"
            << "Was trying to merge objects where one product had been put in the input file and the other had not "
               "been."
            << "\n"
            << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
            << "that need to be merged in the first place.\n";
      }
      if (original->isPresent()) {
        ProductDescription const& desc = productDescription();
        if (mergeableRunProductMetadata == nullptr || desc.branchType() != InRun) {
          original->mergeProduct(iFrom.get());
        } else {
          MergeableRunProductMetadata::MergeDecision decision =
              mergeableRunProductMetadata->getMergeDecision(desc.processName());
          if (decision == MergeableRunProductMetadata::MERGE) {
            original->mergeProduct(iFrom.get());
          } else if (decision == MergeableRunProductMetadata::REPLACE) {
            // Note this swaps the content of the product where the
            // both products branches are present and the product is
            // also present (was put) in the branch. A module might
            // have already gotten a pointer to the product so we
            // keep those pointers valid. This does not call swap
            // on the Wrapper.
            original->swapProduct(iFrom.get());
          }
          // If the decision is IGNORE, do nothing
        }
      }
      // If both have isPresent false, do nothing

    } else if (original->hasIsProductEqual()) {
      if (original->isPresent() && iFrom->isPresent()) {
        if (!original->isProductEqual(iFrom.get())) {
          auto const& bd = productDescription();
          edm::LogError("RunLumiMerging")
              << "ProductResolver::mergeTheProduct\n"
              << "Two run/lumi products for the same run/lumi which should be equal are not\n"
              << "Using the first, ignoring the second\n"
              << "className = " << bd.className() << "\n"
              << "moduleLabel = " << bd.moduleLabel() << "\n"
              << "instance = " << bd.productInstanceName() << "\n"
              << "process = " << bd.processName() << "\n";
        }
      } else if (!original->isPresent() && iFrom->isPresent()) {
        setProduct(std::move(iFrom));
      }
      // if not iFrom->isPresent(), do nothing
    } else {
      auto const& bd = productDescription();
      edm::LogWarning("RunLumiMerging") << "ProductResolver::mergeTheProduct\n"
                                        << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
                                        << "Using the first, ignoring the second in merge\n"
                                        << "className = " << bd.className() << "\n"
                                        << "moduleLabel = " << bd.moduleLabel() << "\n"
                                        << "instance = " << bd.productInstanceName() << "\n"
                                        << "process = " << bd.processName() << "\n";
      if (!original->isPresent() && iFrom->isPresent()) {
        setProduct(std::move(iFrom));
      }
      // In other cases, do nothing
    }
  }

  namespace {
    void extendException(cms::Exception& e, ProductDescription const& bd, ModuleCallingContext const* mcc) {
      e.addContext(std::string("While reading from source ") + bd.className() + " " + bd.moduleLabel() + " '" +
                   bd.productInstanceName() + "' " + bd.processName());
      if (mcc) {
        edm::exceptionContext(e, *mcc);
      }
    }
  }  // namespace
  ProductResolverBase::Resolution DelayedReaderInputProductResolver::resolveProduct_(
      Principal const& principal, SharedResourcesAcquirer*, ModuleCallingContext const* mcc) const {
    return resolveProductImpl<true>([this, &principal, mcc]() {
      auto branchType = principal.branchType();
      if (branchType == InLumi || branchType == InRun) {
        //delayed get has not been allowed with Run or Lumis
        // The file may already be closed so the reader is invalid
        return;
      }
      auto context = mcc;
      if (!context) {
        context = CurrentModuleOnThread::getCurrentModuleOnThread();
      }
      if (context and branchType == InEvent and aux_) {
        aux_->preModuleDelayedGetSignal_.emit(*(context->getStreamContext()), *context);
      }

      auto sentry(make_sentry(context, [this, branchType](ModuleCallingContext const* iContext) {
        if (branchType == InEvent and aux_) {
          aux_->postModuleDelayedGetSignal_.emit(*(iContext->getStreamContext()), *iContext);
        }
      }));

      if (auto reader = principal.reader()) {
        std::unique_lock<std::recursive_mutex> guard;
        if (auto sr = reader->sharedResources().second) {
          guard = std::unique_lock<std::recursive_mutex>(*sr);
        }
        if (not productResolved()) {
          try {
            //another thread could have beaten us here
            setProduct(reader->getProduct(productDescription().branchID(), &principal, context));
          } catch (cms::Exception& e) {
            extendException(e, productDescription(), context);
            throw;
          } catch (std::exception const& e) {
            auto newExcept = edm::Exception(errors::StdException) << e.what();
            extendException(newExcept, productDescription(), context);
            throw newExcept;
          }
        }
      }
    });
  }

  void DelayedReaderInputProductResolver::retrieveAndMerge_(
      Principal const& principal, MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
    if (auto reader = principal.reader()) {
      std::unique_lock<std::recursive_mutex> guard;
      if (auto sr = reader->sharedResources().second) {
        guard = std::unique_lock<std::recursive_mutex>(*sr);
      }

      //Can't use resolveProductImpl since it first checks to see
      // if the product was already retrieved and then returns if it is
      auto edp(reader->getProduct(productDescription().branchID(), &principal));

      if (edp.get() != nullptr) {
        if (edp->isMergeable() && productDescription().branchType() == InRun && !edp->hasSwap()) {
          throw Exception(errors::LogicError)
              << "Missing definition of member function swap for branch name " << productDescription().branchName()
              << "\n"
              << "Mergeable data types written to a Run must have the swap member function defined"
              << "\n";
        }
        if (status() == defaultStatus() || status() == ProductStatus::ProductSet ||
            (status() == ProductStatus::ResolveFailed && !productDescription().isMergeable())) {
          setOrMergeProduct(std::move(edp), mergeableRunProductMetadata);
        } else {  // status() == ResolveFailed && productDescription().isMergeable()
          throw Exception(errors::MismatchedInputFiles)
              << "Merge of Run or Lumi product failed for branch " << productDescription().branchName() << "\n"
              << "The product branch was dropped in the first run or lumi fragment and present in a later one"
              << "\n"
              << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
              << "that need to be merged in the first place.\n";
        }
      } else if (status() == defaultStatus()) {
        setFailedStatus();
      } else if (status() != ProductStatus::ResolveFailed && productDescription().isMergeable()) {
        throw Exception(errors::MismatchedInputFiles)
            << "Merge of Run or Lumi product failed for branch " << productDescription().branchName() << "\n"
            << "The product branch was present in first run or lumi fragment and dropped in a later one"
            << "\n"
            << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
            << "that need to be merged in the first place.\n";
      }
      // Do nothing in other case. status is ResolveFailed already or
      // it is not mergeable and the status is ProductSet
    }
  }

  void MergeableInputProductResolver::setOrMergeProduct(
      std::shared_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
    if (status() == defaultStatus()) {
      //resolveProduct has not been called or it failed
      setProduct(std::move(prod));
    } else {
      mergeProduct(std::move(prod), mergeableRunProductMetadata);
    }
  }

  void DelayedReaderInputProductResolver::setMergeableRunProductMetadata_(MergeableRunProductMetadata const* mrpm) {
    setMergeableRunProductMetadataInProductData(mrpm);
  }

  void DelayedReaderInputProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                         Principal const& principal,
                                                         ServiceToken const& token,
                                                         SharedResourcesAcquirer* sra,
                                                         ModuleCallingContext const* mcc) const noexcept {
    //need to try changing m_prefetchRequested before adding to m_waitingTasks
    bool expected = false;
    bool prefetchRequested = m_prefetchRequested.compare_exchange_strong(expected, true);
    m_waitingTasks.add(waitTask);

    if (prefetchRequested) {
      ServiceWeakToken weakToken = token;
      auto workToDo = [this, mcc, &principal, weakToken]() {
        //need to make sure Service system is activated on the reading thread
        ServiceRegistry::Operate operate(weakToken.lock());
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          resolveProductImpl<true>([this, &principal, mcc]() {
            if (principal.branchType() != InEvent && principal.branchType() != InProcess) {
              return;
            }
            if (auto reader = principal.reader()) {
              std::unique_lock<std::recursive_mutex> guard;
              if (auto sr = reader->sharedResources().second) {
                guard = std::unique_lock<std::recursive_mutex>(*sr);
              }
              if (not productResolved()) {
                try {
                  //another thread could have finished this while we were waiting
                  setProduct(reader->getProduct(productDescription().branchID(), &principal, mcc));
                } catch (cms::Exception& e) {
                  extendException(e, productDescription(), mcc);
                  throw;
                } catch (std::exception const& e) {
                  auto newExcept = edm::Exception(errors::StdException) << e.what();
                  extendException(newExcept, productDescription(), mcc);
                  throw newExcept;
                }
              }
            }
          });
        } catch (...) {
          this->m_waitingTasks.doneWaiting(std::current_exception());
          return;
        }
        this->m_waitingTasks.doneWaiting(nullptr);
      };

      SerialTaskQueueChain* queue = nullptr;
      if (auto reader = principal.reader()) {
        if (auto shared_res = reader->sharedResources().first) {
          queue = &(shared_res->serialQueueChain());
        }
      }
      if (queue) {
        queue->push(*waitTask.group(), workToDo);
      } else {
        //Have to create a new task
        auto t = make_functor_task(workToDo);
        waitTask.group()->run([t]() {
          TaskSentry s{t};
          t->execute();
        });
      }
    }
  }

  void DelayedReaderInputProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      m_prefetchRequested = false;
      m_waitingTasks.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void DelayedReaderInputProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
  }

  bool DelayedReaderInputProductResolver::isFromCurrentProcess() const { return false; }

  void PutOnReadInputProductResolver::putProduct(std::unique_ptr<WrapperBase> edp) const {
    if (status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << productDescription().branchName() << "\n";
    }

    setProduct(std::move(edp));  // ProductResolver takes ownership
  }

  bool PutOnReadInputProductResolver::isFromCurrentProcess() const { return false; }

  ProductResolverBase::Resolution PutOnReadInputProductResolver::resolveProduct_(Principal const&,
                                                                                 SharedResourcesAcquirer*,
                                                                                 ModuleCallingContext const*) const {
    return resolveProductImpl<false>([]() { return; });
  }

  void PutOnReadInputProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                     Principal const& principal,
                                                     ServiceToken const& token,
                                                     SharedResourcesAcquirer* sra,
                                                     ModuleCallingContext const* mcc) const noexcept {}

  void PutOnReadInputProductResolver::putOrMergeProduct(std::unique_ptr<WrapperBase> edp) const {
    setOrMergeProduct(std::move(edp), nullptr);
  }

  ProductResolverBase::Resolution PuttableProductResolver::resolveProduct_(Principal const&,
                                                                           SharedResourcesAcquirer*,
                                                                           ModuleCallingContext const*) const {
    return resolveProductImpl<false>([]() { return; });
  }

  void PuttableProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                               Principal const& principal,
                                               ServiceToken const& token,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const noexcept {
    if (productDescription().branchType() == InProcess &&
        mcc->parent().globalContext()->transition() == GlobalContext::Transition::kAccessInputProcessBlock) {
      // This is an accessInputProcessBlock transition
      // We cannot access produced products in those transitions.
      return;
    }
    if (productDescription().availableOnlyAtEndTransition() and mcc) {
      if (not mcc->parent().isAtEndTransition()) {
        return;
      }
    }

    if (waitingTasks_) {
      // using a waiting task to do a callback guarantees that the
      // waitingTasks_ list (from the worker) will be released from
      // waiting even if the module does not put this data product
      // or the module has an exception while running
      waitingTasks_->add(waitTask);
    }
  }

  void PuttableProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    auto worker = iConfigure.findWorker(productDescription().moduleLabel());
    if (worker) {
      waitingTasks_ = &worker->waitingTaskList();
    }
  }

  void UnscheduledProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
    worker_ = iConfigure.findWorker(productDescription().moduleLabel());
  }

  ProductResolverBase::Resolution UnscheduledProductResolver::resolveProduct_(Principal const&,
                                                                              SharedResourcesAcquirer*,
                                                                              ModuleCallingContext const*) const {
    if (worker_) {
      return resolveProductImpl<false>([] {});
    }
    return Resolution(nullptr);
  }

  void UnscheduledProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                  Principal const& principal,
                                                  ServiceToken const& token,
                                                  SharedResourcesAcquirer* sra,
                                                  ModuleCallingContext const* mcc) const noexcept {
    assert(worker_);
    //need to try changing prefetchRequested_ before adding to waitingTasks_
    bool expected = false;
    bool prefetchRequested = prefetchRequested_.compare_exchange_strong(expected, true);
    waitingTasks_.add(waitTask);
    if (prefetchRequested) {
      //Have to create a new task which will make sure the state for UnscheduledProductResolver
      // is properly set after the module has run
      auto t = make_waiting_task([this](std::exception_ptr const* iPtr) {
        //The exception is being rethrown because resolveProductImpl sets the ProductResolver to a failed
        // state for the case where an exception occurs during the call to the function.
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          resolveProductImpl<true>([iPtr]() {
            if (iPtr) {
              std::rethrow_exception(*iPtr);
            }
          });
        } catch (...) {
          waitingTasks_.doneWaiting(std::current_exception());
          return;
        }
        waitingTasks_.doneWaiting(nullptr);
      });

      ParentContext parentContext(mcc);
      EventTransitionInfo const& info = aux_->eventTransitionInfo();
      worker_->doWorkAsync<OccurrenceTraits<EventPrincipal, TransitionActionStreamBegin> >(
          WaitingTaskHolder(*waitTask.group(), t),
          info,
          token,
          info.principal().streamID(),
          parentContext,
          mcc->getStreamContext());
    }
  }

  void UnscheduledProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      prefetchRequested_ = false;
      waitingTasks_.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void TransformingProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
    worker_ = iConfigure.findWorker(productDescription().moduleLabel());
    // worker can be missing if the corresponding module is
    // unscheduled and none of its products are consumed
    if (worker_) {
      index_ = worker_->transformIndex(productDescription());
    }
  }

  ProductResolverBase::Resolution TransformingProductResolver::resolveProduct_(Principal const&,
                                                                               SharedResourcesAcquirer*,
                                                                               ModuleCallingContext const*) const {
    if (worker_) {
      return resolveProductImpl<false>([] {});
    }
    return Resolution(nullptr);
  }

  void TransformingProductResolver::putProduct(std::unique_ptr<WrapperBase> edp) const {
    // Override putProduct() to not set the resolver status to
    // ResolveFailed when the Event::commit_() checks which produced
    // products were actually produced and which not, because the
    // transforming products are never produced by time of commit_()
    // by construction.
    if (edp) {
      ProducedProductResolver::putProduct(std::move(edp));
    }
  }

  void TransformingProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                   Principal const& principal,
                                                   ServiceToken const& token,
                                                   SharedResourcesAcquirer* sra,
                                                   ModuleCallingContext const* mcc) const noexcept {
    assert(worker_ != nullptr);
    //need to try changing prefetchRequested_ before adding to waitingTasks_
    bool expected = false;
    bool prefetchRequested = prefetchRequested_.compare_exchange_strong(expected, true);
    waitingTasks_.add(waitTask);
    if (prefetchRequested) {
      //Have to create a new task which will make sure the state for TransformingProductResolver
      // is properly set after the module has run
      auto t = make_waiting_task([this](std::exception_ptr const* iPtr) {
        //The exception is being rethrown because resolveProductImpl sets the ProductResolver to a failed
        // state for the case where an exception occurs during the call to the function.
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          resolveProductImpl<true>([iPtr]() {
            if (iPtr) {
              std::rethrow_exception(*iPtr);
            }
          });
        } catch (...) {
          waitingTasks_.doneWaiting(std::current_exception());
          return;
        }
        waitingTasks_.doneWaiting(nullptr);
      });

      //This gives a lifetime greater than this call
      ParentContext parent(mcc);
      mcc_ = ModuleCallingContext(
          worker_->description(), index_ + 1, ModuleCallingContext::State::kPrefetching, parent, nullptr);

      EventTransitionInfo const& info = aux_->eventTransitionInfo();
      worker_->doTransformAsync(WaitingTaskHolder(*waitTask.group(), t),
                                index_,
                                info.principal(),
                                token,
                                info.principal().streamID(),
                                mcc_,
                                mcc->getStreamContext());
    }
  }

  void TransformingProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      prefetchRequested_ = false;
      waitingTasks_.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void ProducedProductResolver::putProduct(std::unique_ptr<WrapperBase> edp) const {
    if (status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << productDescription().branchName() << "\n";
    }

    setProduct(std::move(edp));  // ProductResolver takes ownership
  }

  bool ProducedProductResolver::isFromCurrentProcess() const { return true; }

  void DataManagingProductResolver::connectTo(ProductResolverBase const& iOther, Principal const*) { assert(false); }

  void DataManagingProductResolver::checkType(WrapperBase const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if (typeID != TypeID{productDescription().unwrappedType().unvalidatedTypeInfo()}) {
      // Types do not match.
      throw Exception(errors::EventCorruption)
          << "Product on branch " << productDescription().branchName() << " is of wrong type.\n"
          << "It is supposed to be of type " << productDescription().className() << ".\n"
          << "It is actually of type " << typeID.className() << ".\n";
    }
  }

  void DataManagingProductResolver::setProduct(std::unique_ptr<WrapperBase> edp) const {
    if (edp) {
      checkType(*edp);
      productData_.unsafe_setWrapper(std::move(edp));
      theStatus_ = ProductStatus::ProductSet;
    } else {
      setFailedStatus();
    }
  }
  void DataManagingProductResolver::setProduct(std::shared_ptr<WrapperBase> edp) const {
    if (edp) {
      checkType(*edp);
      productData_.unsafe_setWrapper(std::move(edp));
      theStatus_ = ProductStatus::ProductSet;
    } else {
      setFailedStatus();
    }
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool DataManagingProductResolver::productUnavailable_() const {
    auto presentStatus = status();
    if (presentStatus == ProductStatus::ProductSet) {
      return !(getProductData().wrapper()->isPresent());
    }
    return presentStatus != ProductStatus::ResolveNotRun;
  }

  bool DataManagingProductResolver::productResolved_() const {
    auto s = status();
    return (s != defaultStatus()) or (s == ProductStatus::ProductDeleted);
  }

  // This routine returns true if the product was deleted early in order to save memory
  bool DataManagingProductResolver::productWasDeleted_() const { return status() == ProductStatus::ProductDeleted; }

  bool DataManagingProductResolver::productWasFetchedAndIsValid_() const {
    if (status() == ProductStatus::ProductSet) {
      if (getProductData().wrapper()->isPresent()) {
        return true;
      }
    }
    return false;
  }

  void DataManagingProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) {
    productData_.setProvenance(provRetriever);
  }

  void DataManagingProductResolver::setProductID_(ProductID const& pid) { productData_.setProductID(pid); }

  void DataManagingProductResolver::setMergeableRunProductMetadataInProductData(
      MergeableRunProductMetadata const* mrpm) {
    productData_.setMergeableRunProductMetadata(mrpm);
  }

  ProductProvenance const* DataManagingProductResolver::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void DataManagingProductResolver::resetProductData_(bool deleteEarly) {
    if (theStatus_ == ProductStatus::ProductSet) {
      productData_.resetProductData();
    }
    if (deleteEarly) {
      theStatus_ = ProductStatus::ProductDeleted;
    } else {
      resetStatus();
    }
  }

  bool DataManagingProductResolver::singleProduct_() const { return true; }

  void AliasProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) {
    realProduct_.setProductProvenanceRetriever(provRetriever);
  }

  void AliasProductResolver::setProductID_(ProductID const& pid) { realProduct_.setProductID(pid); }

  ProductProvenance const* AliasProductResolver::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void AliasProductResolver::resetProductData_(bool deleteEarly) { realProduct_.resetProductData_(deleteEarly); }

  bool AliasProductResolver::singleProduct_() const { return true; }

}  // namespace edm
