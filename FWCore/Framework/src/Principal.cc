/**----------------------------------------------------------------------
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Principal.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "ProductResolvers.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TClass.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
#include <atomic>

namespace edm {

  static ProcessHistory const s_emptyProcessHistory;

  static
  void
  maybeThrowMissingDictionaryException(TypeID const& productType, bool isElement, std::vector<TypeID> const& missingDictionaries) {
    if(isElement) {
      TypeSet missingTypes;
      if(binary_search_all(missingDictionaries, productType)) {
        checkTypeDictionary(productType,missingTypes);
        throwMissingDictionariesException(missingTypes);
      }
    } else {
      TClass* cl = TClass::GetClass(wrappedClassName(productType.className()).c_str());
      TypeID wrappedProductType = TypeID(cl->GetTypeInfo());
      TypeSet missingTypes;
      if(binary_search_all(missingDictionaries, wrappedProductType)) {
        checkClassDictionary(wrappedProductType,missingTypes);
        throwMissingDictionariesException(missingTypes);
      }
    }
  }

  static
  void
  throwProductNotFoundException(char const* where, errors::ErrorCodes error, BranchID const& bid) {
    throw Exception(error, "InvalidID")
      << "Principal::" << where << ": no product with given branch id: "<< bid << "\n";
  }

  static
  std::shared_ptr<cms::Exception>
  makeNotFoundException(char const* where, KindOfType kindOfType,
                        TypeID const& productType, std::string const& label, std::string const& instance, std::string const& process) {
    std::shared_ptr<cms::Exception> exception = std::make_shared<Exception>(errors::ProductNotFound);
    if (kindOfType == PRODUCT_TYPE) {
      *exception << "Principal::" << where << ": Found zero products matching all criteria\nLooking for type: " << productType << "\n"
                 << "Looking for module label: " << label << "\n" << "Looking for productInstanceName: " << instance << "\n"
                 << (process.empty() ? "" : "Looking for process: ") << process << "\n";
    } else {
      *exception << "Principal::" << where << ": Found zero products matching all criteria\nLooking for a container with elements of type: " << productType << "\n"
                 << "Looking for module label: " << label << "\n" << "Looking for productInstanceName: " << instance << "\n"
                 << (process.empty() ? "" : "Looking for process: ") << process << "\n";
    }
    return exception;
  }

  static
  void
  throwAmbiguousException(const char* where, TypeID const& productType,std::string const& label, std::string const& instance, std::string const& process) {
    cms::Exception exception("AmbiguousProduct");
    exception << "Principal::" << where << ": More than 1 product matches all criteria\nLooking for type: " << productType << "\n"
    << "Looking for module label: " << label << "\n" << "Looking for productInstanceName: " << instance << "\n"
    << (process.empty() ? "" : "Looking for process: ") << process << "\n"
    << "This can only occur with get function calls using a Handle<View> argument.\n"
    << "Try a get not using a View or change the instance name of one of the products";
    throw exception;
    
  }

  namespace {
    void failedToRegisterConsumesMany(edm::TypeID const& iType) {
      cms::Exception exception("GetManyWithoutRegistration");
      exception << "::getManyByType called for " << iType
                << " without a corresponding consumesMany being called for this module. \n";
      throw exception;
    }
    
    void failedToRegisterConsumes(KindOfType kindOfType,
                                  TypeID const& productType,
                                  std::string const& moduleLabel,
                                  std::string const& productInstanceName,
                                  std::string const& processName) {
      cms::Exception exception("GetByLabelWithoutRegistration");
      exception << "::getByLabel without corresponding call to consumes or mayConsumes for this module.\n"
                << (kindOfType == PRODUCT_TYPE ? "  type: " : " type: edm::View<") << productType
                << (kindOfType == PRODUCT_TYPE ? "\n  module label: " : ">\n  module label: ") << moduleLabel
                <<"\n  product instance name: '" << productInstanceName
                <<"'\n  process name: '" << processName << "'\n";
      throw exception;
    }
}

  //0 means unset
  static std::atomic<Principal::CacheIdentifier_t> s_nextIdentifier{1};
  static inline Principal::CacheIdentifier_t nextIdentifier() {
    return s_nextIdentifier.fetch_add(1,std::memory_order_acq_rel);
  }
  
  Principal::Principal(std::shared_ptr<ProductRegistry const> reg,
                       std::shared_ptr<ProductResolverIndexHelper const> productLookup,
                       ProcessConfiguration const& pc,
                       BranchType bt,
                       HistoryAppender* historyAppender,
                       bool isForPrimaryProcess) :
    EDProductGetter(),
    processHistoryPtr_(),
    processHistoryID_(),
    processConfiguration_(&pc),
    productResolvers_(),
    preg_(reg),
    productLookup_(productLookup),
    lookupProcessOrder_(productLookup->lookupProcessNames().size(), 0),
    reader_(),
    branchType_(bt),
    historyAppender_(historyAppender),
    cacheIdentifier_(nextIdentifier())
  {
    productResolvers_.resize(reg->getNextIndexValue(bt));
    //Now that these have been set, we can create the list of Branches we need.
    std::string const source("source");
    ProductRegistry::ProductList const& prodsList = reg->productList();
    // The constructor of an alias product holder takes as an argument the product holder for which it is an alias.
    // So, the non-alias product holders must be created first.
    // Therefore, on this first pass, skip current EDAliases.
    bool hasAliases = false;
    for(auto const& prod : prodsList) {
      BranchDescription const& bd = prod.second;
      if(bd.branchType() == branchType_) {
        if(isForPrimaryProcess or bd.processName() == pc.processName()) {
          if(bd.isAlias()) {
            hasAliases = true;
          } else {
            auto cbd = std::make_shared<BranchDescription const>(bd);
            if(bd.produced()) {
              if(bd.moduleLabel() == source) {
                addSourceProduct(cbd);
              } else if(bd.onDemand()) {
                assert(branchType_ == InEvent);
                addUnscheduledProduct(cbd);
              } else {
                addScheduledProduct(cbd);
              }
            } else {
              addInputProduct(cbd);
            }
          }
        } else {
          //We are in a SubProcess and this branch is from the parent
          auto cbd =std::make_shared<BranchDescription const>(bd);
          addParentProcessProduct(cbd);
        }
      }
    }
    // Now process any EDAliases
    if(hasAliases) {
      for(auto const& prod : prodsList) {
        BranchDescription const& bd = prod.second;
        if(bd.isAlias() && bd.branchType() == branchType_) {
          auto cbd = std::make_shared<BranchDescription const>(bd);
          addAliasedProduct(cbd);
        }
      }
    }

    // Now create the ProductResolvers that search in reverse process
    // order and are used for queries where the process name is the
    // empty string
    std::vector<std::string> const& lookupProcessNames = productLookup_->lookupProcessNames();
    std::vector<ProductResolverIndex> matchingHolders(lookupProcessNames.size(), ProductResolverIndexInvalid);
    std::vector<bool> ambiguous(lookupProcessNames.size(), false);
    unsigned int beginElements = productLookup_->beginElements();
    std::vector<TypeID> const& sortedTypeIDs = productLookup_->sortedTypeIDs();
    std::vector<ProductResolverIndexHelper::Range> const& ranges = productLookup_->ranges();
    std::vector<ProductResolverIndexHelper::IndexAndNames> const& indexAndNames = productLookup_->indexAndNames();
    std::vector<char> const& processNamesCharArray = productLookup_->processNames();

    if (!sortedTypeIDs.empty()) {
      ProductResolverIndex productResolverIndex = ProductResolverIndexInvalid;
      for(unsigned int k = 0, kEnd = sortedTypeIDs.size(); k < kEnd; ++k) {
        ProductResolverIndexHelper::Range const& range = ranges.at(k);
        for (unsigned int i = range.begin(); i < range.end(); ++i) {
          ProductResolverIndexHelper::IndexAndNames const& product = indexAndNames.at(i);
          if (product.startInProcessNames() == 0) {
            if (productResolverIndex != ProductResolverIndexInvalid) {
              std::shared_ptr<ProductResolverBase> newHolder = std::make_shared<NoProcessProductResolver>(matchingHolders, ambiguous);
              productResolvers_.at(productResolverIndex) = newHolder;
              matchingHolders.assign(lookupProcessNames.size(), ProductResolverIndexInvalid);
              ambiguous.assign(lookupProcessNames.size(), false);
            }
            productResolverIndex = product.index();
          } else {
            std::string process(&processNamesCharArray.at(product.startInProcessNames()));
            auto iter = std::find(lookupProcessNames.begin(), lookupProcessNames.end(), process);
            assert(iter != lookupProcessNames.end());
            ProductResolverIndex iMatchingIndex = product.index();
            assert(iMatchingIndex != ProductResolverIndexInvalid);
            if (iMatchingIndex == ProductResolverIndexAmbiguous) {
              assert(k >= beginElements);
              ambiguous.at(iter - lookupProcessNames.begin()) = true;
            } else {
              matchingHolders.at(iter - lookupProcessNames.begin()) = iMatchingIndex;
            }
          }
        }
      }
      std::shared_ptr<ProductResolverBase> newHolder = std::make_shared<NoProcessProductResolver>(matchingHolders, ambiguous);
      productResolvers_.at(productResolverIndex) = newHolder;
    }
  }

  Principal::~Principal() {
  }

  // Number of products in the Principal.
  // For products in an input file and not yet read in due to delayed read,
  // this routine assumes a real product is there.
  size_t
  Principal::size() const {
    size_t size = 0U;
    for(auto const& prod : *this) {
      if(prod->singleProduct() && // Not a NoProcessProductResolver
         !prod->productUnavailable() &&
         !prod->unscheduledWasNotRun() &&
         !prod->branchDescription().dropped()) {
        ++size;
      }
    }
    return size;
  }

  // adjust provenance for input products after new input file has been merged
  bool
  Principal::adjustToNewProductRegistry(ProductRegistry const& reg) {
    ProductRegistry::ProductList const& prodsList = reg.productList();
    for(auto const& prod : prodsList) {
      BranchDescription const& bd = prod.second;
      if(!bd.produced() && (bd.branchType() == branchType_)) {
        auto cbd = std::make_shared<BranchDescription const>(bd);
        auto phb = getExistingProduct(cbd->branchID());
        if(phb == nullptr || phb->branchDescription().branchName() != cbd->branchName()) {
            return false;
        }
        phb->resetBranchDescription(cbd);
      }
    }
    return true;
  }

  void
  Principal::addScheduledProduct(std::shared_ptr<BranchDescription const> bd) {
    auto phb = std::make_unique<PuttableProductResolver>(std::move(bd));
    addProductOrThrow(std::move(phb));
  }

  void
  Principal::addSourceProduct(std::shared_ptr<BranchDescription const> bd) {
    auto phb = std::make_unique<PuttableProductResolver>(std::move(bd));
    addProductOrThrow(std::move(phb));
  }

  void
  Principal::addInputProduct(std::shared_ptr<BranchDescription const> bd) {
    addProductOrThrow(std::make_unique<InputProductResolver>(std::move(bd)));
  }

  void
  Principal::addUnscheduledProduct(std::shared_ptr<BranchDescription const> bd) {
    addProductOrThrow(std::make_unique<UnscheduledProductResolver>(std::move(bd)));
  }

  void
  Principal::addAliasedProduct(std::shared_ptr<BranchDescription const> bd) {
    ProductResolverIndex index = preg_->indexFrom(bd->originalBranchID());
    assert(index != ProductResolverIndexInvalid);

    addProductOrThrow(std::make_unique<AliasProductResolver>(std::move(bd), dynamic_cast<ProducedProductResolver&>(*productResolvers_[index])));
  }

  void
  Principal::addParentProcessProduct(std::shared_ptr<BranchDescription const> bd) {
    addProductOrThrow(std::make_unique<ParentProcessProductResolver>(std::move(bd)));
  }

  // "Zero" the principal so it can be reused for another Event.
  void
  Principal::clearPrincipal() {
    processHistoryPtr_.reset();
    processHistoryID_ = ProcessHistoryID();
    reader_ = nullptr;
    for(auto& prod : *this) {
      prod->resetProductData();
    }
  }

  void
  Principal::deleteProduct(BranchID const& id) const {
    auto phb = getExistingProduct(id);
    assert(nullptr != phb);
    phb->unsafe_deleteProduct();
  }
  
  // Set the principal for the Event, Lumi, or Run.
  void
  Principal::fillPrincipal(ProcessHistoryID const& hist,
                           ProcessHistoryRegistry const& processHistoryRegistry,
                           DelayedReader* reader) {
    //increment identifier here since clearPrincipal isn't called for Run/Lumi
    cacheIdentifier_=nextIdentifier();
    if(reader) {
      reader_ = reader;
    }

    if (historyAppender_ && productRegistry().anyProductProduced()) {
      processHistoryPtr_ =
        historyAppender_->appendToProcessHistory(hist,
                                                 processHistoryRegistry.getMapped(hist),
                                                 *processConfiguration_);
      processHistoryID_ = processHistoryPtr_->id();
    }
    else {
      std::shared_ptr<ProcessHistory const> inputProcessHistory;
      if (hist.isValid()) {
        //does not own the pointer
        auto noDel =[](void const*){};
        inputProcessHistory =
        std::shared_ptr<ProcessHistory const>(processHistoryRegistry.getMapped(hist),noDel);
        if (inputProcessHistory.get() == nullptr) {
          throw Exception(errors::LogicError)
            << "Principal::fillPrincipal\n"
            << "Input ProcessHistory not found in registry\n"
            << "Contact a Framework developer\n";
        }
      } else {
        //Since this is static we don't want it deleted
        inputProcessHistory = std::shared_ptr<ProcessHistory const>(&s_emptyProcessHistory,[](void const*){});
      }
      processHistoryID_ = hist;
      processHistoryPtr_ = inputProcessHistory;        
    }

    if (orderProcessHistoryID_ != processHistoryID_) {
      std::vector<std::string> const& lookupProcessNames = productLookup_->lookupProcessNames();
      lookupProcessOrder_.assign(lookupProcessNames.size(), 0);
      unsigned int k = 0;
      for (auto iter = processHistoryPtr_->rbegin(),
                iEnd = processHistoryPtr_->rend();
           iter != iEnd; ++iter) {
        auto nameIter = std::find(lookupProcessNames.begin(), lookupProcessNames.end(), iter->processName());
        if (nameIter == lookupProcessNames.end()) {
          continue;
        }
        lookupProcessOrder_.at(k) = nameIter - lookupProcessNames.begin();
        ++k;
      }
      orderProcessHistoryID_ = processHistoryID_;
    }
  }

  ProductResolverBase*
  Principal::getExistingProduct(BranchID const& branchID) {
    return const_cast<ProductResolverBase*>( const_cast<const Principal*>(this)->getExistingProduct(branchID));
  }

  ProductResolverBase const*
  Principal::getExistingProduct(BranchID const& branchID) const {
    ProductResolverIndex index = preg_->indexFrom(branchID);
    assert(index != ProductResolverIndexInvalid);
    return productResolvers_.at(index).get();
  }

  ProductResolverBase const*
  Principal::getExistingProduct(ProductResolverBase const& productResolver) const {
    auto phb = getExistingProduct(productResolver.branchDescription().branchID());
    if(nullptr != phb && BranchKey(productResolver.branchDescription()) != BranchKey(phb->branchDescription())) {
      BranchDescription const& newProduct = phb->branchDescription();
      BranchDescription const& existing = productResolver.branchDescription();
      if(newProduct.branchName() != existing.branchName() && newProduct.branchID() == existing.branchID()) {
        throw cms::Exception("HashCollision") << "Principal::getExistingProduct\n" <<
          " Branch " << newProduct.branchName() << " has same branch ID as branch " << existing.branchName() << "\n" <<
          "Workaround: change process name or product instance name of " << newProduct.branchName() << "\n";
      } else {
        assert(nullptr == phb || BranchKey(productResolver.branchDescription()) == BranchKey(phb->branchDescription()));
      }
    }
    return phb;
  }

  void
  Principal::addProduct_(std::unique_ptr<ProductResolverBase> productResolver) {
    BranchDescription const& bd = productResolver->branchDescription();
    assert (!bd.className().empty());
    assert (!bd.friendlyClassName().empty());
    assert (!bd.moduleLabel().empty());
    assert (!bd.processName().empty());
    SharedProductPtr phb(productResolver.release());

    ProductResolverIndex index = preg_->indexFrom(bd.branchID());
    assert(index != ProductResolverIndexInvalid);
    productResolvers_[index] = phb;
  }

  void
  Principal::addProductOrThrow(std::unique_ptr<ProductResolverBase> productResolver) {
    ProductResolverBase const* phb = getExistingProduct(*productResolver);
    if(phb != nullptr) {
      BranchDescription const& bd = productResolver->branchDescription();
      throw Exception(errors::InsertFailure, "AlreadyPresent")
          << "addProductOrThrow: Problem found while adding product, "
          << "product already exists for ("
          << bd.friendlyClassName() << ","
          << bd.moduleLabel() << ","
          << bd.productInstanceName() << ","
          << bd.processName()
          << ")\n";
    }
    addProduct_(std::move(productResolver));
  }

  Principal::ConstProductResolverPtr
  Principal::getProductResolver(BranchID const& bid) const {
    ProductResolverIndex index = preg_->indexFrom(bid);
    if(index == ProductResolverIndexInvalid){
       return ConstProductResolverPtr();
    }
    return getProductResolverByIndex(index);
  }

  Principal::ConstProductResolverPtr
  Principal::getProductResolverByIndex(ProductResolverIndex const& index) const {

    ConstProductResolverPtr const phb = productResolvers_[index].get();
    return phb;
  }

  BasicHandle
  Principal::getByLabel(KindOfType kindOfType,
                        TypeID const& typeID,
                        InputTag const& inputTag,
                        EDConsumerBase const* consumer,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const {

    ProductData const* result = findProductByLabel(kindOfType, typeID, inputTag, consumer, sra, mcc);
    if(result == 0) {
      return BasicHandle(makeHandleExceptionFactory([=]()->std::shared_ptr<cms::Exception> {
        return makeNotFoundException("getByLabel", kindOfType, typeID, inputTag.label(), inputTag.instance(), inputTag.process());
      }));
    }
    return BasicHandle(result->wrapper(), &(result->provenance()));
  }

  BasicHandle
  Principal::getByLabel(KindOfType kindOfType,
                        TypeID const& typeID,
                        std::string const& label,
                        std::string const& instance,
                        std::string const& process,
                        EDConsumerBase const* consumer,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const {

    ProductData const* result = findProductByLabel(kindOfType, typeID, label, instance, process,consumer, sra, mcc);
    if(result == 0) {
      return BasicHandle(makeHandleExceptionFactory([=]()->std::shared_ptr<cms::Exception> {
        return makeNotFoundException("getByLabel", kindOfType, typeID, label, instance, process);
      }));
    }
    return BasicHandle(result->wrapper(), &(result->provenance()));
  }

  BasicHandle
  Principal::getByToken(KindOfType,
                        TypeID const&,
                        ProductResolverIndex index,
                        bool skipCurrentProcess,
                        bool& ambiguous,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const {
    assert(index !=ProductResolverIndexInvalid);
    auto& productResolver = productResolvers_[index];
    assert(0!=productResolver.get());
    ProductResolverBase::ResolveStatus resolveStatus;
    ProductData const* productData = productResolver->resolveProduct(resolveStatus, *this, skipCurrentProcess, sra, mcc);
    if(resolveStatus == ProductResolverBase::Ambiguous) {
      ambiguous = true;
      return BasicHandle();
    }
    if(productData == 0) {
      return BasicHandle();
    }
    return BasicHandle(productData->wrapper(), &(productData->provenance()));
  }

  void
  Principal::prefetch(ProductResolverIndex index,
                      bool skipCurrentProcess,
                      ModuleCallingContext const* mcc) const {
    auto const& productResolver = productResolvers_.at(index);
    assert(0!=productResolver.get());
    ProductResolverBase::ResolveStatus resolveStatus;
    productResolver->resolveProduct(resolveStatus, *this,skipCurrentProcess, nullptr, mcc);
  }

  void
  Principal::getManyByType(TypeID const& typeID,
                           BasicHandleVec& results,
                           EDConsumerBase const* consumer,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const {

    assert(results.empty());

    if(unlikely(consumer and (not consumer->registeredToConsumeMany(typeID,branchType())))) {
      failedToRegisterConsumesMany(typeID);
    }

    // This finds the indexes to all the ProductResolver's matching the type
    ProductResolverIndexHelper::Matches matches =
      productLookup().relatedIndexes(PRODUCT_TYPE, typeID);

    if (matches.numberOfMatches() == 0) {
      maybeThrowMissingDictionaryException(typeID, false, preg_->missingDictionaries());
      return;
    }

    results.reserve(matches.numberOfMatches());

    // Loop over the ProductResolvers. Add the products that are actually
    // present into the results. This will also trigger delayed reading,
    // on demand production, and check for deleted products as appropriate.

    // Over the years the code that uses getManyByType has grown to depend
    // on the ordering of the results. The order originally was just an
    // accident of how previous versions of the code were written, but
    // here we have to go through some extra effort to preserve that ordering.

    // We build a list of holders that match a particular label and instance.
    // When that list is complete we call findProducts, which loops over
    // that list in reverse order of the ProcessHistory (starts with the
    // most recent).  Then we clear the list and repeat this until all the
    // matching label and instance subsets have been dealt with.

    // Note that the function isFullyResolved returns true for the ProductResolvers
    // that are associated with an empty process name. Those are the ones that
    // know how to search for the most recent process name matching
    // a label and instance. We do not need these for getManyByType and
    // skip them. In addition to skipping them, we make use of the fact
    // that they mark the beginning of each subset of holders with the same
    // label and instance. They tell us when to call findProducts. 

    std::vector<ProductResolverBase const*> holders;

    for(unsigned int i = 0; i < matches.numberOfMatches(); ++i) {

      ProductResolverIndex index = matches.index(i);

      if(!matches.isFullyResolved(i)) {
        if(!holders.empty()) {
          // Process the ones with a particular module label and instance
          findProducts(holders, typeID, results, sra, mcc);
          holders.clear();
        }
      } else {
        ProductResolverBase const* productResolver = productResolvers_.at(index).get();
        assert(productResolver);
        holders.push_back(productResolver);
      }
    }
    // Do not miss the last subset of products
    if(!holders.empty()) {
      findProducts(holders, typeID, results, sra, mcc);
    }
    return;
  }

  void
  Principal::findProducts(std::vector<ProductResolverBase const*> const& holders,
                          TypeID const&,
                          BasicHandleVec& results,
                          SharedResourcesAcquirer* sra,
                          ModuleCallingContext const* mcc) const {

    for (auto iter = processHistoryPtr_->rbegin(),
              iEnd = processHistoryPtr_->rend();
         iter != iEnd; ++iter) {
      std::string const& process = iter->processName();
      for (auto productResolver : holders) {
        BranchDescription const& bd = productResolver->branchDescription();
        if (process == bd.processName()) {

          // Ignore aliases to avoid matching the same product multiple times.
          if(bd.isAlias()) {
            continue;
          }

          ProductResolverBase::ResolveStatus resolveStatus;
          ProductData const* productData = productResolver->resolveProduct(resolveStatus, *this,false, sra, mcc);
          if(productData) {
            // Skip product if not available.
            results.emplace_back(productData->wrapper(), &(productData->provenance()));
          }
        }
      }
    }
  }

  ProductData const*
  Principal::findProductByLabel(KindOfType kindOfType,
                                TypeID const& typeID,
                                InputTag const& inputTag,
                                EDConsumerBase const* consumer,
                                SharedResourcesAcquirer* sra,
                                ModuleCallingContext const* mcc) const {

    bool skipCurrentProcess = inputTag.willSkipCurrentProcess();

    ProductResolverIndex index = inputTag.indexFor(typeID, branchType(), &productRegistry());

    if (index == ProductResolverIndexInvalid) {

      char const* processName = inputTag.process().c_str();
      if (skipCurrentProcess) {
        processName = "\0";
      }

      index = productLookup().index(kindOfType,
                                    typeID,
                                    inputTag.label().c_str(),
                                    inputTag.instance().c_str(),
                                    processName);

      if(index == ProductResolverIndexAmbiguous) {
        throwAmbiguousException("findProductByLabel", typeID, inputTag.label(), inputTag.instance(), inputTag.process());
      } else if (index == ProductResolverIndexInvalid) {
        ProductResolverIndexHelper::Matches matches =
          productLookup().relatedIndexes(kindOfType, typeID);

        if (matches.numberOfMatches() == 0) {
          maybeThrowMissingDictionaryException(typeID, kindOfType == ELEMENT_TYPE, preg_->missingDictionaries());
        }
        return 0;
      }
      inputTag.tryToCacheIndex(index, typeID, branchType(), &productRegistry());
    }
    if(unlikely( consumer and (not consumer->registeredToConsume(index, skipCurrentProcess, branchType())))) {
      failedToRegisterConsumes(kindOfType,typeID,inputTag.label(),inputTag.instance(),inputTag.process());
    }

    
    auto const& productResolver = productResolvers_[index];

    ProductResolverBase::ResolveStatus resolveStatus;
    ProductData const* productData = productResolver->resolveProduct(resolveStatus, *this, skipCurrentProcess, sra, mcc);
    if(resolveStatus == ProductResolverBase::Ambiguous) {
      throwAmbiguousException("findProductByLabel", typeID, inputTag.label(), inputTag.instance(), inputTag.process());
    }
    return productData;
  }

  ProductData const*
  Principal::findProductByLabel(KindOfType kindOfType,
                                TypeID const& typeID,
                                std::string const& label,
                                std::string const& instance,
                                std::string const& process,
                                EDConsumerBase const* consumer,
                                SharedResourcesAcquirer* sra,
                                ModuleCallingContext const* mcc) const {

    ProductResolverIndex index = productLookup().index(kindOfType,
                                                     typeID,
                                                     label.c_str(),
                                                     instance.c_str(),
                                                     process.c_str());
   
    if(index == ProductResolverIndexAmbiguous) {
      throwAmbiguousException("findProductByLabel", typeID, label, instance, process);
    } else if (index == ProductResolverIndexInvalid) {
      ProductResolverIndexHelper::Matches matches =
        productLookup().relatedIndexes(kindOfType, typeID);

      if (matches.numberOfMatches() == 0) {
        maybeThrowMissingDictionaryException(typeID, kindOfType == ELEMENT_TYPE, preg_->missingDictionaries());
      }
      return 0;
    }
    
    if(unlikely( consumer and (not consumer->registeredToConsume(index, false, branchType())))) {
      failedToRegisterConsumes(kindOfType,typeID,label,instance,process);
    }
    
    auto const& productResolver = productResolvers_[index];

    ProductResolverBase::ResolveStatus resolveStatus;
    ProductData const* productData = productResolver->resolveProduct(resolveStatus, *this, false, sra, mcc);
    if(resolveStatus == ProductResolverBase::Ambiguous) {
      throwAmbiguousException("findProductByLabel", typeID, label, instance, process);
    }
    return productData;
  }

  ProductData const*
  Principal::findProductByTag(TypeID const& typeID, InputTag const& tag, ModuleCallingContext const* mcc) const {
    ProductData const* productData =
      findProductByLabel(PRODUCT_TYPE,
                         typeID,
                         tag,
                         nullptr,
                         nullptr,
                         mcc);
    return productData;
  }

  Provenance
  Principal::getProvenance(BranchID const& bid,
                           ModuleCallingContext const* mcc) const {
    ConstProductResolverPtr const phb = getProductResolver(bid);
    if(phb == nullptr) {
      throwProductNotFoundException("getProvenance", errors::ProductNotFound, bid);
    }

    if(phb->unscheduledWasNotRun()) {
      ProductResolverBase::ResolveStatus status;
      if(not phb->resolveProduct(status,*this,false, nullptr, mcc) ) {
        throwProductNotFoundException("getProvenance(onDemand)", errors::ProductNotFound, bid);
      }
    }
    return *phb->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  Principal::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provenances.clear();
    for(auto const& productResolver : *this) {
      if(productResolver->singleProduct() && productResolver->provenanceAvailable() && !productResolver->branchDescription().isAlias()) {
        // We do not attempt to get the event/lumi/run status from the provenance,
        // because the per event provenance may have been dropped.
        if(productResolver->provenance()->branchDescription().present()) {
           provenances.push_back(productResolver->provenance());
        }
      }
    }
  }

  // This one is also mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance for dropped branches.
  void
  Principal::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    provenances.clear();
    for(auto const& productResolver : *this) {
      if(productResolver->singleProduct() && !productResolver->branchDescription().isAlias()) {
        if(productResolver->stableProvenance()->branchDescription().present()) {
           provenances.push_back(productResolver->stableProvenance());
        }
      }
    }
  }
 
  void
  Principal::recombine(Principal& other, std::vector<BranchID> const& bids) {
    for(auto& prod : bids) {
      ProductResolverIndex index= preg_->indexFrom(prod);
      assert(index!=ProductResolverIndexInvalid);
      ProductResolverIndex indexO = other.preg_->indexFrom(prod);
      assert(indexO!=ProductResolverIndexInvalid);
      get_underlying_safe(productResolvers_[index]).swap(get_underlying_safe(other.productResolvers_[indexO]));
    }
    reader_->mergeReaders(other.reader());
  }

  WrapperBase const*
  Principal::getIt(ProductID const&) const {
    assert(false);
    return nullptr;
  }

  WrapperBase const*
  Principal::getThinnedProduct(ProductID const&, unsigned int&) const {
    assert(false);
    return nullptr;
  }

  void
  Principal::getThinnedProducts(ProductID const&,
                                  std::vector<WrapperBase const*>&,
                                  std::vector<unsigned int>&) const {
    assert(false);
  }

  void
  Principal::putOrMerge(std::unique_ptr<WrapperBase> prod, ProductResolverBase const* phb) const {
    phb->putOrMergeProduct(std::move(prod));
  }

  void
  Principal::putOrMerge(BranchDescription const& bd, std::unique_ptr<WrapperBase>  edp) const {
    if(edp.get() == nullptr) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
      << "put: Cannot put because unique_ptr to product is null."
      << "\n";
    }
    auto phb = getExistingProduct(bd.branchID());
    assert(phb);
    // ProductResolver assumes ownership
    putOrMerge(std::move(edp), phb);
  }


  void
  Principal::adjustIndexesAfterProductRegistryAddition() {
    if(preg_->getNextIndexValue(branchType_) != productResolvers_.size()) {
      productResolvers_.resize(preg_->getNextIndexValue(branchType_));
      for(auto const& prod : preg_->productList()) {
        BranchDescription const& bd = prod.second;
        if(bd.branchType() == branchType_) {
          ProductResolverIndex index = preg_->indexFrom(bd.branchID());
          assert(index != ProductResolverIndexInvalid);
          if(!productResolvers_[index]) {
            // no product holder.  Must add one. The new entry must be an input product holder.
            assert(!bd.produced());
            auto cbd = std::make_shared<BranchDescription const>(bd);
            addInputProduct(cbd);
          }
        }
      }
    }
    assert(preg_->getNextIndexValue(branchType_) == productResolvers_.size());
  }
  
  void
  Principal::readAllFromSourceAndMergeImmediately() {
    for(auto & prod : *this) {
      ProductResolverBase & phb = *prod;
      if(phb.singleProduct() && !phb.branchDescription().produced()) {
        if(!phb.productUnavailable()) {
          resolveProductImmediately(phb);
        }
      }
    }
  }
  void
  Principal::resolveProductImmediately(ProductResolverBase& phb)  {
    if(phb.branchDescription().produced()) return; // nothing to do.
    if(!reader()) return; // nothing to do.
    
    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(phb.branchDescription());
    std::unique_ptr<WrapperBase> edp(reader()->getProduct(bk, this));
    
    // Now fix up the ProductResolver
    if(edp.get() != nullptr) {
      putOrMerge(std::move(edp), &phb);
    }
  }

}
