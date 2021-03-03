/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "ProductResolvers.h"
#include "Worker.h"
#include "UnscheduledAuxiliary.h"
#include "UnscheduledConfigurator.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/src/ProductDeletedException.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <cassert>
#include <utility>

static constexpr unsigned int kUnsetOffset = 0;
static constexpr unsigned int kAmbiguousOffset = 1;
static constexpr unsigned int kMissingOffset = 2;

namespace edm {

  void DataManagingProductResolver::throwProductDeletedException() const {
    ProductDeletedException exception;
    exception << "DataManagingProductResolver::resolveProduct_: The product matching all criteria was already deleted\n"
              << "Looking for type: " << branchDescription().unwrappedTypeID() << "\n"
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

  void DataManagingProductResolver::mergeProduct(std::unique_ptr<WrapperBase> iFrom,
                                                 MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
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
            << "Merge of Run or Lumi product failed for branch " << branchDescription().branchName() << "\n"
            << "Was trying to merge objects where one product had been put in the input file and the other had not "
               "been."
            << "\n"
            << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
            << "that need to be merged in the first place.\n";
      }
      if (original->isPresent()) {
        BranchDescription const& desc = branchDescription_();
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
          auto const& bd = branchDescription();
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
      auto const& bd = branchDescription();
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

  ProductResolverBase::Resolution InputProductResolver::resolveProduct_(Principal const& principal,
                                                                        bool,
                                                                        SharedResourcesAcquirer*,
                                                                        ModuleCallingContext const* mcc) const {
    return resolveProductImpl<true>([this, &principal, mcc]() {
      auto branchType = principal.branchType();
      if (branchType == InLumi || branchType == InRun) {
        //delayed get has not been allowed with Run or Lumis
        // The file may already be closed so the reader is invalid
        return;
      }
      if (mcc and (branchType == InEvent || branchType == InProcess) and aux_) {
        aux_->preModuleDelayedGetSignal_.emit(*(mcc->getStreamContext()), *mcc);
      }

      auto sentry(make_sentry(mcc, [this, branchType](ModuleCallingContext const* iContext) {
        if ((branchType == InEvent || branchType == InProcess) and aux_) {
          aux_->postModuleDelayedGetSignal_.emit(*(iContext->getStreamContext()), *iContext);
        }
      }));

      if (auto reader = principal.reader()) {
        std::unique_lock<std::recursive_mutex> guard;
        if (auto sr = reader->sharedResources().second) {
          guard = std::unique_lock<std::recursive_mutex>(*sr);
        }
        if (not productResolved()) {
          //another thread could have beaten us here
          putProduct(reader->getProduct(branchDescription().branchID(), &principal, mcc));
        }
      }
    });
  }

  void InputProductResolver::retrieveAndMerge_(Principal const& principal,
                                               MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
    if (auto reader = principal.reader()) {
      std::unique_lock<std::recursive_mutex> guard;
      if (auto sr = reader->sharedResources().second) {
        guard = std::unique_lock<std::recursive_mutex>(*sr);
      }

      //Can't use resolveProductImpl since it first checks to see
      // if the product was already retrieved and then returns if it is
      std::unique_ptr<WrapperBase> edp(reader->getProduct(branchDescription().branchID(), &principal));

      if (edp.get() != nullptr) {
        if (edp->isMergeable() && branchDescription().branchType() == InRun && !edp->hasSwap()) {
          throw Exception(errors::LogicError)
              << "Missing definition of member function swap for branch name " << branchDescription().branchName()
              << "\n"
              << "Mergeable data types written to a Run must have the swap member function defined"
              << "\n";
        }
        if (status() == defaultStatus() || status() == ProductStatus::ProductSet ||
            (status() == ProductStatus::ResolveFailed && !branchDescription().isMergeable())) {
          putOrMergeProduct(std::move(edp), mergeableRunProductMetadata);
        } else {  // status() == ResolveFailed && branchDescription().isMergeable()
          throw Exception(errors::MismatchedInputFiles)
              << "Merge of Run or Lumi product failed for branch " << branchDescription().branchName() << "\n"
              << "The product branch was dropped in the first run or lumi fragment and present in a later one"
              << "\n"
              << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
              << "that need to be merged in the first place.\n";
        }
      } else if (status() == defaultStatus()) {
        setFailedStatus();
      } else if (status() != ProductStatus::ResolveFailed && branchDescription().isMergeable()) {
        throw Exception(errors::MismatchedInputFiles)
            << "Merge of Run or Lumi product failed for branch " << branchDescription().branchName() << "\n"
            << "The product branch was present in first run or lumi fragment and dropped in a later one"
            << "\n"
            << "The solution is to drop the branch on input. Or better do not create inconsistent files\n"
            << "that need to be merged in the first place.\n";
      }
      // Do nothing in other case. status is ResolveFailed already or
      // it is not mergeable and the status is ProductSet
    }
  }

  void InputProductResolver::setMergeableRunProductMetadata_(MergeableRunProductMetadata const* mrpm) {
    setMergeableRunProductMetadataInProductData(mrpm);
  }

  void InputProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                            Principal const& principal,
                                            bool skipCurrentProcess,
                                            ServiceToken const& token,
                                            SharedResourcesAcquirer* sra,
                                            ModuleCallingContext const* mcc) const {
    //need to try changing m_prefetchRequested before adding to m_waitingTasks
    bool expected = false;
    bool prefetchRequested = m_prefetchRequested.compare_exchange_strong(expected, true);
    m_waitingTasks.add(waitTask);

    if (prefetchRequested) {
      auto workToDo = [this, mcc, &principal, token]() {
        //need to make sure Service system is activated on the reading thread
        ServiceRegistry::Operate operate(token);
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          resolveProductImpl<true>([this, &principal, mcc]() {
            if (principal.branchType() != InEvent) {
              return;
            }
            if (auto reader = principal.reader()) {
              std::unique_lock<std::recursive_mutex> guard;
              if (auto sr = reader->sharedResources().second) {
                guard = std::unique_lock<std::recursive_mutex>(*sr);
              }
              if (not productResolved()) {
                //another thread could have finished this while we were waiting
                putProduct(reader->getProduct(branchDescription().branchID(), &principal, mcc));
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
        queue->push(workToDo);
      } else {
        //Have to create a new task
        auto t = make_functor_task(tbb::task::allocate_root(), workToDo);
        tbb::task::spawn(*t);
      }
    }
  }

  void InputProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      m_prefetchRequested = false;
      m_waitingTasks.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void InputProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
  }

  bool InputProductResolver::isFromCurrentProcess() const { return false; }

  ProductResolverBase::Resolution PuttableProductResolver::resolveProduct_(Principal const&,
                                                                           bool skipCurrentProcess,
                                                                           SharedResourcesAcquirer*,
                                                                           ModuleCallingContext const*) const {
    if (!skipCurrentProcess) {
      //'false' means never call the lambda function
      return resolveProductImpl<false>([]() { return; });
    }
    return Resolution(nullptr);
  }

  void PuttableProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               ServiceToken const& token,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const {
    if (not skipCurrentProcess) {
      if (branchDescription().availableOnlyAtEndTransition() and mcc) {
        if (not mcc->parent().isAtEndTransition()) {
          return;
        }
      }
      //Need to try modifying prefetchRequested_ before adding to m_waitingTasks
      bool expected = false;
      bool prefetchRequested = prefetchRequested_.compare_exchange_strong(expected, true);
      m_waitingTasks.add(waitTask);

      if (worker_ and prefetchRequested) {
        //using a waiting task to do a callback guarantees that
        // the m_waitingTasks list will be released from waiting even
        // if the module does not put this data product or the
        // module has an exception while running

        auto waiting = make_waiting_task(tbb::task::allocate_root(), [this](std::exception_ptr const* iException) {
          if (nullptr != iException) {
            m_waitingTasks.doneWaiting(*iException);
          } else {
            m_waitingTasks.doneWaiting(std::exception_ptr());
          }
        });
        worker_->callWhenDoneAsync(WaitingTaskHolder(waiting));
      }
    }
  }

  void PuttableProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    ProducedProductResolver::putProduct_(std::move(edp));
    bool expected = false;
    if (prefetchRequested_.compare_exchange_strong(expected, true)) {
      m_waitingTasks.doneWaiting(std::exception_ptr());
    }
  }

  void PuttableProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      prefetchRequested_ = false;
      m_waitingTasks.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void PuttableProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    worker_ = iConfigure.findWorker(branchDescription().moduleLabel());
  }

  void UnscheduledProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
    worker_ = iConfigure.findWorker(branchDescription().moduleLabel());
  }

  ProductResolverBase::Resolution UnscheduledProductResolver::resolveProduct_(Principal const&,
                                                                              bool skipCurrentProcess,
                                                                              SharedResourcesAcquirer*,
                                                                              ModuleCallingContext const*) const {
    if (!skipCurrentProcess and worker_) {
      return resolveProductImpl<true>([this]() {
        edm::Exception ex(errors::UnimplementedFeature);
        ex << "Attempting to run unscheduled module without doing prefetching";
        std::ostringstream ost;
        ost << "Calling produce method for unscheduled module " << worker_->description()->moduleName() << "/'"
            << worker_->description()->moduleLabel() << "'";
        ex.addContext(ost.str());
        throw ex;
      });
    }
    return Resolution(nullptr);
  }

  void UnscheduledProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                  Principal const& principal,
                                                  bool skipCurrentProcess,
                                                  ServiceToken const& token,
                                                  SharedResourcesAcquirer* sra,
                                                  ModuleCallingContext const* mcc) const {
    if (skipCurrentProcess) {
      return;
    }
    if (worker_ == nullptr) {
      throw cms::Exception("LogicError") << "UnscheduledProductResolver::prefetchAsync_()  called with null worker_. "
                                            "This should not happen, please contact framework developers.";
    }
    //need to try changing prefetchRequested_ before adding to waitingTasks_
    bool expected = false;
    bool prefetchRequested = prefetchRequested_.compare_exchange_strong(expected, true);
    waitingTasks_.add(waitTask);
    if (prefetchRequested) {
      //Have to create a new task which will make sure the state for UnscheduledProductResolver
      // is properly set after the module has run
      auto t = make_waiting_task(tbb::task::allocate_root(), [this](std::exception_ptr const* iPtr) {
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
      worker_->doWorkAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> >(
          WaitingTaskHolder(t), info, token, info.principal().streamID(), parentContext, mcc->getStreamContext());
    }
  }

  void UnscheduledProductResolver::resetProductData_(bool deleteEarly) {
    if (not deleteEarly) {
      prefetchRequested_ = false;
      waitingTasks_.reset();
    }
    DataManagingProductResolver::resetProductData_(deleteEarly);
  }

  void ProducedProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if (status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }

    setProduct(std::move(edp));  // ProductResolver takes ownership
  }

  void InputProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if (not productResolved()) {
      //Another thread could have set this
      setProduct(std::move(edp));
    }
  }

  bool ProducedProductResolver::isFromCurrentProcess() const { return true; }

  void DataManagingProductResolver::connectTo(ProductResolverBase const& iOther, Principal const*) { assert(false); }

  void DataManagingProductResolver::putOrMergeProduct_(
      std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
    if (not prod) {
      return;
    }
    if (status() == defaultStatus()) {
      //resolveProduct has not been called or it failed
      putProduct(std::move(prod));
    } else {
      mergeProduct(std::move(prod), mergeableRunProductMetadata);
    }
  }

  void DataManagingProductResolver::checkType(WrapperBase const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if (typeID != TypeID{branchDescription().unwrappedType().unvalidatedTypeInfo()}) {
      // Types do not match.
      throw Exception(errors::EventCorruption)
          << "Product on branch " << branchDescription().branchName() << " is of wrong type.\n"
          << "It is supposed to be of type " << branchDescription().className() << ".\n"
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

  bool DataManagingProductResolver::productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const {
    if (iSkipCurrentProcess and isFromCurrentProcess()) {
      return false;
    }
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

  void AliasProductResolver::putProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
        << "AliasProductResolver::putProduct_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void AliasProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                                MergeableRunProductMetadata const*) const {
    throw Exception(errors::LogicError)
        << "AliasProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, MergeableRunProductMetadata "
           "const*) not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  SwitchBaseProductResolver::SwitchBaseProductResolver(std::shared_ptr<BranchDescription const> bd,
                                                       DataManagingOrAliasProductResolver& realProduct)
      : realProduct_(realProduct), productData_(std::move(bd)), prefetchRequested_(false) {
    // Parentage of this branch is always the same by construction, so we can compute the ID just "once" here.
    Parentage p;
    p.setParents(std::vector<BranchID>{realProduct.branchDescription().originalBranchID()});
    parentageID_ = p.id();
    ParentageRegistry::instance()->insertMapped(p);
  }

  void SwitchBaseProductResolver::connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) {
    throw Exception(errors::LogicError)
        << "SwitchBaseProductResolver::connectTo() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void SwitchBaseProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    worker_ = iConfigure.findWorker(branchDescription().moduleLabel());
  }

  ProductResolverBase::Resolution SwitchBaseProductResolver::resolveProductImpl(Resolution res) const {
    if (res.data() == nullptr)
      return res;
    return Resolution(&productData_);
  }

  bool SwitchBaseProductResolver::productResolved_() const {
    // SwitchProducer will never put anything in the event, and
    // "false" will make Event::commit_() to call putProduct() with
    // null unique_ptr<WrapperBase> to signal that the produce() was
    // run.
    return false;
  }

  void SwitchBaseProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                                     MergeableRunProductMetadata const*) const {
    throw Exception(errors::LogicError)
        << "SwitchBaseProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, "
           "MergeableRunProductMetadata const*) not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void SwitchBaseProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) {
    productData_.setProvenance(provRetriever);
  }

  void SwitchBaseProductResolver::setProductID_(ProductID const& pid) {
    // insertIntoSet is const, so let's exploit that to fake the getting of the "input" product
    productData_.setProductID(pid);
  }

  void SwitchBaseProductResolver::resetProductData_(bool deleteEarly) {
    productData_.resetProductData();
    realProduct_.resetProductData_(deleteEarly);
    if (not deleteEarly) {
      prefetchRequested_ = false;
      waitingTasks_.reset();
    }
  }

  void SwitchBaseProductResolver::unsafe_setWrapperAndProvenance() const {
    // update provenance
    productData_.provenance().store()->insertIntoSet(ProductProvenance(branchDescription().branchID(), parentageID_));
    // Use the Wrapper of the pointed-to resolver, but the provenance of this resolver
    productData_.unsafe_setWrapper(realProduct().getProductData().sharedConstWrapper());
  }

  SwitchProducerProductResolver::SwitchProducerProductResolver(std::shared_ptr<BranchDescription const> bd,
                                                               DataManagingOrAliasProductResolver& realProduct)
      : SwitchBaseProductResolver(std::move(bd), realProduct), status_(defaultStatus_) {}

  ProductResolverBase::Resolution SwitchProducerProductResolver::resolveProduct_(Principal const& principal,
                                                                                 bool skipCurrentProcess,
                                                                                 SharedResourcesAcquirer* sra,
                                                                                 ModuleCallingContext const* mcc) const {
    if (status_ == ProductStatus::ResolveFailed) {
      return resolveProductImpl(realProduct().resolveProduct(principal, skipCurrentProcess, sra, mcc));
    }
    return Resolution(nullptr);
  }

  void SwitchProducerProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                     Principal const& principal,
                                                     bool skipCurrentProcess,
                                                     ServiceToken const& token,
                                                     SharedResourcesAcquirer* sra,
                                                     ModuleCallingContext const* mcc) const {
    if (skipCurrentProcess) {
      return;
    }
    if (branchDescription().availableOnlyAtEndTransition() and mcc and not mcc->parent().isAtEndTransition()) {
      return;
    }

    //need to try changing prefetchRequested before adding to waitingTasks
    bool expected = false;
    bool doPrefetchRequested = prefetchRequested().compare_exchange_strong(expected, true);
    waitingTasks().add(waitTask);

    if (doPrefetchRequested) {
      //using a waiting task to do a callback guarantees that
      // the waitingTasks() list will be released from waiting even
      // if the module does not put this data product or the
      // module has an exception while running
      auto waiting = make_waiting_task(tbb::task::allocate_root(), [this](std::exception_ptr const* iException) {
        if (nullptr != iException) {
          waitingTasks().doneWaiting(*iException);
        } else {
          unsafe_setWrapperAndProvenance();
          waitingTasks().doneWaiting(std::exception_ptr());
        }
      });
      worker()->callWhenDoneAsync(WaitingTaskHolder(waiting));
    }
  }

  void SwitchProducerProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if (status_ != defaultStatus_) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product for a branch " << branchDescription().branchName()
          << "This makes no sense for SwitchProducerProductResolver.\nContact a Framework developer";
    }
    // Let's use ResolveFailed to signal that produce() was called, as
    // there is no real product in this resolver
    status_ = ProductStatus::ResolveFailed;
    bool expected = false;
    if (prefetchRequested().compare_exchange_strong(expected, true)) {
      unsafe_setWrapperAndProvenance();
      waitingTasks().doneWaiting(std::exception_ptr());
    }
  }

  bool SwitchProducerProductResolver::productUnavailable_() const {
    // if produce() was run (ResolveFailed), ask from the real resolver
    if (status_ == ProductStatus::ResolveFailed) {
      return realProduct().productUnavailable();
    }
    return true;
  }

  void SwitchProducerProductResolver::resetProductData_(bool deleteEarly) {
    SwitchBaseProductResolver::resetProductData_(deleteEarly);
    if (not deleteEarly) {
      status_ = defaultStatus_;
    }
  }

  ProductResolverBase::Resolution SwitchAliasProductResolver::resolveProduct_(Principal const& principal,
                                                                              bool skipCurrentProcess,
                                                                              SharedResourcesAcquirer* sra,
                                                                              ModuleCallingContext const* mcc) const {
    return resolveProductImpl(realProduct().resolveProduct(principal, skipCurrentProcess, sra, mcc));
  }

  void SwitchAliasProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                  Principal const& principal,
                                                  bool skipCurrentProcess,
                                                  ServiceToken const& token,
                                                  SharedResourcesAcquirer* sra,
                                                  ModuleCallingContext const* mcc) const {
    if (skipCurrentProcess) {
      return;
    }

    //need to try changing prefetchRequested_ before adding to waitingTasks_
    bool expected = false;
    bool doPrefetchRequested = prefetchRequested().compare_exchange_strong(expected, true);
    waitingTasks().add(waitTask);

    if (doPrefetchRequested) {
      //using a waiting task to do a callback guarantees that
      // the waitingTasks() list will be released from waiting even
      // if the module does not put this data product or the
      // module has an exception while running
      auto waiting = make_waiting_task(tbb::task::allocate_root(), [this](std::exception_ptr const* iException) {
        if (nullptr != iException) {
          waitingTasks().doneWaiting(*iException);
        } else {
          unsafe_setWrapperAndProvenance();
          waitingTasks().doneWaiting(std::exception_ptr());
        }
      });
      realProduct().prefetchAsync(WaitingTaskHolder(waiting), principal, skipCurrentProcess, token, sra, mcc);
    }
  }

  void SwitchAliasProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
        << "SwitchAliasProductResolver::putProduct() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void ParentProcessProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) {
    provRetriever_ = provRetriever;
  }

  void ParentProcessProductResolver::setProductID_(ProductID const&) {}

  ProductProvenance const* ParentProcessProductResolver::productProvenancePtr_() const {
    return provRetriever_ ? provRetriever_->branchIDToProvenance(bd_->originalBranchID()) : nullptr;
  }

  void ParentProcessProductResolver::resetProductData_(bool deleteEarly) {}

  bool ParentProcessProductResolver::singleProduct_() const { return true; }

  void ParentProcessProductResolver::putProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
        << "ParentProcessProductResolver::putProduct_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void ParentProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                                        MergeableRunProductMetadata const*) const {
    throw Exception(errors::LogicError)
        << "ParentProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, "
           "MergeableRunProductMetadata const*) not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void ParentProcessProductResolver::throwNullRealProduct() const {
    // In principle, this ought to be fixed. I noticed one hits this error
    // when in a SubProcess and calling the Event::getProvenance function
    // with a BranchID to a branch from an earlier SubProcess or the top
    // level process and this branch is not kept in this SubProcess. It might
    // be possible to hit this in other contexts. I say it ought to be
    // fixed because one does not encounter this issue if the SubProcesses
    // are split into genuinely different processes (in principle that
    // ought to give identical behavior and results). No user has ever
    // reported this issue which has been around for some time and it was only
    // noticed when testing some rare corner cases after modifying Core code.
    // After discussing this with Chris we decided that at least for the moment
    // there are higher priorities than fixing this ... I converted it so it
    // causes an exception instead of a seg fault. The issue that may need to
    // be addressed someday is how ProductResolvers for non-kept branches are
    // connected to earlier SubProcesses.
    throw Exception(errors::LogicError)
        << "ParentProcessProductResolver::throwNullRealProduct RealProduct pointer not set in this context.\n"
        << "Contact a Framework developer\n";
  }

  NoProcessProductResolver::NoProcessProductResolver(std::vector<ProductResolverIndex> const& matchingHolders,
                                                     std::vector<bool> const& ambiguous,
                                                     bool madeAtEnd)
      : matchingHolders_(matchingHolders),
        ambiguous_(ambiguous),
        lastCheckIndex_(ambiguous_.size() + kUnsetOffset),
        lastSkipCurrentCheckIndex_(lastCheckIndex_.load()),
        prefetchRequested_(false),
        skippingPrefetchRequested_(false),
        madeAtEnd_{madeAtEnd} {
    assert(ambiguous_.size() == matchingHolders_.size());
  }

  ProductResolverBase::Resolution NoProcessProductResolver::tryResolver(unsigned int index,
                                                                        Principal const& principal,
                                                                        bool skipCurrentProcess,
                                                                        SharedResourcesAcquirer* sra,
                                                                        ModuleCallingContext const* mcc) const {
    ProductResolverBase const* productResolver = principal.getProductResolverByIndex(matchingHolders_[index]);
    return productResolver->resolveProduct(principal, skipCurrentProcess, sra, mcc);
  }

  ProductResolverBase::Resolution NoProcessProductResolver::resolveProduct_(Principal const& principal,
                                                                            bool skipCurrentProcess,
                                                                            SharedResourcesAcquirer* sra,
                                                                            ModuleCallingContext const* mcc) const {
    //See if we've already cached which Resolver we should call or if
    // we know it is ambiguous
    const unsigned int choiceSize = ambiguous_.size();

    //madeAtEnd_==true and not at end transition is the same as skipping the current process
    if ((not skipCurrentProcess) and (madeAtEnd_ and mcc)) {
      skipCurrentProcess = not mcc->parent().isAtEndTransition();
    }

    unsigned int checkCacheIndex = skipCurrentProcess ? lastSkipCurrentCheckIndex_.load() : lastCheckIndex_.load();
    if (checkCacheIndex != choiceSize + kUnsetOffset) {
      if (checkCacheIndex == choiceSize + kAmbiguousOffset) {
        return ProductResolverBase::Resolution::makeAmbiguous();
      } else if (checkCacheIndex == choiceSize + kMissingOffset) {
        return Resolution(nullptr);
      }
      return tryResolver(checkCacheIndex, principal, skipCurrentProcess, sra, mcc);
    }

    std::atomic<unsigned int>& updateCacheIndex = skipCurrentProcess ? lastSkipCurrentCheckIndex_ : lastCheckIndex_;

    std::vector<unsigned int> const& lookupProcessOrder = principal.lookupProcessOrder();
    for (unsigned int k : lookupProcessOrder) {
      assert(k < ambiguous_.size());
      if (k == 0)
        break;  // Done
      if (ambiguous_[k]) {
        updateCacheIndex = choiceSize + kAmbiguousOffset;
        return ProductResolverBase::Resolution::makeAmbiguous();
      }
      if (matchingHolders_[k] != ProductResolverIndexInvalid) {
        auto resolution = tryResolver(k, principal, skipCurrentProcess, sra, mcc);
        if (resolution.data() != nullptr) {
          updateCacheIndex = k;
          return resolution;
        }
      }
    }

    updateCacheIndex = choiceSize + kMissingOffset;
    return Resolution(nullptr);
  }

  void NoProcessProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                Principal const& principal,
                                                bool skipCurrentProcess,
                                                ServiceToken const& token,
                                                SharedResourcesAcquirer* sra,
                                                ModuleCallingContext const* mcc) const {
    bool timeToMakeAtEnd = true;
    if (madeAtEnd_ and mcc) {
      timeToMakeAtEnd = mcc->parent().isAtEndTransition();
    }

    //If timeToMakeAtEnd is false, then it is equivalent to skipping the current process
    if (not skipCurrentProcess and timeToMakeAtEnd) {
      //need to try changing prefetchRequested_ before adding to waitingTasks_
      bool expected = false;
      bool prefetchRequested = prefetchRequested_.compare_exchange_strong(expected, true);
      waitingTasks_.add(waitTask);

      if (prefetchRequested) {
        //we are the first thread to request
        tryPrefetchResolverAsync(0, principal, false, sra, mcc, token);
      }
    } else {
      skippingWaitingTasks_.add(waitTask);
      bool expected = false;
      if (skippingPrefetchRequested_.compare_exchange_strong(expected, true)) {
        //we are the first thread to request
        tryPrefetchResolverAsync(0, principal, true, sra, mcc, token);
      }
    }
  }

  void NoProcessProductResolver::setCache(bool iSkipCurrentProcess,
                                          ProductResolverIndex iIndex,
                                          std::exception_ptr iExceptPtr) const {
    if (not iSkipCurrentProcess) {
      lastCheckIndex_ = iIndex;
      waitingTasks_.doneWaiting(iExceptPtr);
    } else {
      lastSkipCurrentCheckIndex_ = iIndex;
      skippingWaitingTasks_.doneWaiting(iExceptPtr);
    }
  }

  namespace {
    class TryNextResolverWaitingTask : public edm::WaitingTask {
    public:
      TryNextResolverWaitingTask(NoProcessProductResolver const* iResolver,
                                 unsigned int iResolverIndex,
                                 Principal const* iPrincipal,
                                 SharedResourcesAcquirer* iSRA,
                                 ModuleCallingContext const* iMCC,
                                 bool iSkipCurrentProcess,
                                 ServiceToken iToken)
          : resolver_(iResolver),
            principal_(iPrincipal),
            sra_(iSRA),
            mcc_(iMCC),
            serviceToken_(iToken),
            index_(iResolverIndex),
            skipCurrentProcess_(iSkipCurrentProcess) {}

      tbb::task* execute() override {
        auto exceptPtr = exceptionPtr();
        if (exceptPtr) {
          resolver_->prefetchFailed(index_, *principal_, skipCurrentProcess_, *exceptPtr);
        } else {
          if (not resolver_->dataValidFromResolver(index_, *principal_, skipCurrentProcess_)) {
            resolver_->tryPrefetchResolverAsync(
                index_ + 1, *principal_, skipCurrentProcess_, sra_, mcc_, serviceToken_);
          }
        }
        return nullptr;
      }

    private:
      NoProcessProductResolver const* resolver_;
      Principal const* principal_;
      SharedResourcesAcquirer* sra_;
      ModuleCallingContext const* mcc_;
      ServiceToken serviceToken_;
      unsigned int index_;
      bool skipCurrentProcess_;
    };
  }  // namespace

  void NoProcessProductResolver::prefetchFailed(unsigned int iProcessingIndex,
                                                Principal const& principal,
                                                bool iSkipCurrentProcess,
                                                std::exception_ptr iExceptPtr) const {
    std::vector<unsigned int> const& lookupProcessOrder = principal.lookupProcessOrder();
    auto k = lookupProcessOrder[iProcessingIndex];

    setCache(iSkipCurrentProcess, k, iExceptPtr);
  }

  bool NoProcessProductResolver::dataValidFromResolver(unsigned int iProcessingIndex,
                                                       Principal const& principal,
                                                       bool iSkipCurrentProcess) const {
    std::vector<unsigned int> const& lookupProcessOrder = principal.lookupProcessOrder();
    auto k = lookupProcessOrder[iProcessingIndex];
    ProductResolverBase const* productResolver = principal.getProductResolverByIndex(matchingHolders_[k]);

    if (productResolver->productWasFetchedAndIsValid(iSkipCurrentProcess)) {
      setCache(iSkipCurrentProcess, k, nullptr);
      return true;
    }
    return false;
  }

  void NoProcessProductResolver::tryPrefetchResolverAsync(unsigned int iProcessingIndex,
                                                          Principal const& principal,
                                                          bool skipCurrentProcess,
                                                          SharedResourcesAcquirer* sra,
                                                          ModuleCallingContext const* mcc,
                                                          ServiceToken token) const {
    std::vector<unsigned int> const& lookupProcessOrder = principal.lookupProcessOrder();
    auto index = iProcessingIndex;

    const unsigned int choiceSize = ambiguous_.size();
    unsigned int newCacheIndex = choiceSize + kMissingOffset;
    while (index < lookupProcessOrder.size()) {
      auto k = lookupProcessOrder[index];
      if (k == 0) {
        break;
      }
      assert(k < ambiguous_.size());
      if (ambiguous_[k]) {
        newCacheIndex = choiceSize + kAmbiguousOffset;
        break;
      }
      if (matchingHolders_[k] != ProductResolverIndexInvalid) {
        //make new task

        auto task = new (tbb::task::allocate_root())
            TryNextResolverWaitingTask(this, index, &principal, sra, mcc, skipCurrentProcess, token);
        WaitingTaskHolder hTask(task);
        ProductResolverBase const* productResolver = principal.getProductResolverByIndex(matchingHolders_[k]);

        //Make sure the Services are available on this thread
        ServiceRegistry::Operate guard(token);

        productResolver->prefetchAsync(hTask, principal, skipCurrentProcess, token, sra, mcc);
        return;
      }
      ++index;
    }
    //data product unavailable
    setCache(skipCurrentProcess, newCacheIndex, nullptr);
  }

  void NoProcessProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const*) {}

  void NoProcessProductResolver::setProductID_(ProductID const&) {}

  ProductProvenance const* NoProcessProductResolver::productProvenancePtr_() const { return nullptr; }

  inline unsigned int NoProcessProductResolver::unsetIndexValue() const { return ambiguous_.size() + kUnsetOffset; }

  void NoProcessProductResolver::resetProductData_(bool) {
    // This function should never receive 'true'. On the other hand,
    // nothing should break if a 'true' is passed, because
    // NoProcessProductResolver just forwards the resolve
    const auto resetValue = unsetIndexValue();
    lastCheckIndex_ = resetValue;
    lastSkipCurrentCheckIndex_ = resetValue;
    prefetchRequested_ = false;
    skippingPrefetchRequested_ = false;
    waitingTasks_.reset();
    skippingWaitingTasks_.reset();
  }

  bool NoProcessProductResolver::singleProduct_() const { return false; }

  bool NoProcessProductResolver::unscheduledWasNotRun_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::unscheduledWasNotRun_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productUnavailable_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::productUnavailable_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productResolved_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::productResolved_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productWasDeleted_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::productWasDeleted_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productWasFetchedAndIsValid_(bool /*iSkipCurrentProcess*/) const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::productWasFetchedAndIsValid_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::putProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::putProduct_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                                    MergeableRunProductMetadata const*) const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, MergeableRunProductMetadata "
           "const*) not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  BranchDescription const& NoProcessProductResolver::branchDescription_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::branchDescription_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::resetBranchDescription_(std::shared_ptr<BranchDescription const>) {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::resetBranchDescription_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  Provenance const* NoProcessProductResolver::provenance_() const {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::provenance_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::connectTo(ProductResolverBase const&, Principal const*) {
    throw Exception(errors::LogicError)
        << "NoProcessProductResolver::connectTo() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  //---- SingleChoiceNoProcessProductResolver ----------------
  ProductResolverBase::Resolution SingleChoiceNoProcessProductResolver::resolveProduct_(
      Principal const& principal,
      bool skipCurrentProcess,
      SharedResourcesAcquirer* sra,
      ModuleCallingContext const* mcc) const {
    //NOTE: Have to lookup the other ProductResolver each time rather than cache
    // it's pointer since it appears the pointer can change at some later stage
    return principal.getProductResolverByIndex(realResolverIndex_)
        ->resolveProduct(principal, skipCurrentProcess, sra, mcc);
  }

  void SingleChoiceNoProcessProductResolver::prefetchAsync_(WaitingTaskHolder waitTask,
                                                            Principal const& principal,
                                                            bool skipCurrentProcess,
                                                            ServiceToken const& token,
                                                            SharedResourcesAcquirer* sra,
                                                            ModuleCallingContext const* mcc) const {
    principal.getProductResolverByIndex(realResolverIndex_)
        ->prefetchAsync(waitTask, principal, skipCurrentProcess, token, sra, mcc);
  }

  void SingleChoiceNoProcessProductResolver::setProductProvenanceRetriever_(ProductProvenanceRetriever const*) {}

  void SingleChoiceNoProcessProductResolver::setProductID_(ProductID const&) {}

  ProductProvenance const* SingleChoiceNoProcessProductResolver::productProvenancePtr_() const { return nullptr; }

  void SingleChoiceNoProcessProductResolver::resetProductData_(bool) {}

  bool SingleChoiceNoProcessProductResolver::singleProduct_() const { return false; }

  bool SingleChoiceNoProcessProductResolver::unscheduledWasNotRun_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::unscheduledWasNotRun_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool SingleChoiceNoProcessProductResolver::productUnavailable_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::productUnavailable_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool SingleChoiceNoProcessProductResolver::productResolved_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::productResolved_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool SingleChoiceNoProcessProductResolver::productWasDeleted_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::productWasDeleted_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  bool SingleChoiceNoProcessProductResolver::productWasFetchedAndIsValid_(bool /*iSkipCurrentProcess*/) const {
    throw Exception(errors::LogicError) << "SingleChoiceNoProcessProductResolver::productWasFetchedAndIsValid_() not "
                                           "implemented and should never be called.\n"
                                        << "Contact a Framework developer\n";
  }

  void SingleChoiceNoProcessProductResolver::putProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::putProduct_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void SingleChoiceNoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                                                MergeableRunProductMetadata const*) const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, "
           "MergeableRunProductMetadata const*) not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  BranchDescription const& SingleChoiceNoProcessProductResolver::branchDescription_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::branchDescription_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void SingleChoiceNoProcessProductResolver::resetBranchDescription_(std::shared_ptr<BranchDescription const>) {
    throw Exception(errors::LogicError) << "SingleChoiceNoProcessProductResolver::resetBranchDescription_() not "
                                           "implemented and should never be called.\n"
                                        << "Contact a Framework developer\n";
  }

  Provenance const* SingleChoiceNoProcessProductResolver::provenance_() const {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::provenance_() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

  void SingleChoiceNoProcessProductResolver::connectTo(ProductResolverBase const&, Principal const*) {
    throw Exception(errors::LogicError)
        << "SingleChoiceNoProcessProductResolver::connectTo() not implemented and should never be called.\n"
        << "Contact a Framework developer\n";
  }

}  // namespace edm
