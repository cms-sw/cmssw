/**----------------------------------------------------------------------
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Principal.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

//using boost::lambda::_1;

namespace edm {

  ProcessHistory const Principal::emptyProcessHistory_;

  static
  void
  maybeThrowMissingDictionaryException(TypeID const& productType, bool isElement, std::vector<std::string> const& missingDictionaries) {
    if(binary_search_all(missingDictionaries, productType.className())) {
      checkDictionaries(isElement ? productType.className() : wrappedClassName(productType.className()), false);
      throwMissingDictionariesException();
    }
  }

  static
  void
  throwProductNotFoundException(char const* where, errors::ErrorCodes error, BranchID const& bid) {
    throw Exception(error, "InvalidID")
      << "Principal::" << where << ": no product with given branch id: "<< bid << "\n";
  }

  static
  void
  throwCorruptionException(char const* where, std::string const& branchName) {
    throw Exception(errors::EventCorruption)
       << "Principal::" << where <<": Product on branch " << branchName << " occurs twice in the same event.\n";
  }

  static
  boost::shared_ptr<cms::Exception>
  makeNotFoundException(char const* where, TypeID const& productType, std::string const& label, std::string const& instance, std::string const& process) {
    boost::shared_ptr<cms::Exception> exception(new Exception(errors::ProductNotFound));
    *exception << "Principal::" << where << ": Found zero products matching all criteria\nLooking for type: " << productType << "\n"
               << "Looking for module label: " << label << "\n" << "Looking for productInstanceName: " << instance << "\n"
               << (process.empty() ? "" : "Looking for process: ") << process << "\n";
    return exception;
  }

  static
  void
  throwProductDeletedException(const char* where, TypeID const& productType,std::string const& label, std::string const& instance, std::string const& process) {
    boost::shared_ptr<cms::Exception> exception(new ProductDeletedException());
    *exception << "Principal::" << where << ": The product matching all criteria\nLooking for type: " << productType << "\n"
    << "Looking for module label: " << label << "\n" << "Looking for productInstanceName: " << instance << "\n"
    << (process.empty() ? "" : "Looking for process: ") << process << "\n"
    << "Was already deleted. This means there is a configuration error.\nThe module which is asking for this data must be configured to state that it will read this data.";
    throw exception;
    
  }

  
  static
  void
  throwNotFoundException(char const* where, TypeID const& productType, InputTag const& tag) {
    boost::shared_ptr<cms::Exception> exception = makeNotFoundException(where, productType, tag.label(), tag.instance(), tag.process());
    throw *exception;
  }
  

  Principal::Principal(boost::shared_ptr<ProductRegistry const> reg,
                       ProcessConfiguration const& pc,
                       BranchType bt,
                       HistoryAppender* historyAppender) :
    EDProductGetter(),
    processHistoryPtr_(nullptr),
    processHistoryID_(),
    processConfiguration_(&pc),
    productHolders_(reg->constProductList().size(), SharedProductPtr()),
    preg_(reg),
    reader_(),
    productPtrs_(),
    branchType_(bt),
    historyAppender_(historyAppender) {

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
        if(bd.isAlias()) {
          hasAliases = true;
        } else {
          boost::shared_ptr<ConstBranchDescription> cbd(new ConstBranchDescription(bd));
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
      }
    }
    // Now process any EDAliases
    if(hasAliases) {
      for(auto const& prod : prodsList) {
        BranchDescription const& bd = prod.second;
        if(bd.isAlias() && bd.branchType() == branchType_) {
          boost::shared_ptr<ConstBranchDescription> cbd(new ConstBranchDescription(bd));
          addAliasedProduct(cbd);
        }
      }
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
      if(!prod->productUnavailable() && !prod->onDemand() && !prod->branchDescription().dropped()) {
        ++size;
      }
    }
    return size;
  }

  // adjust provenance for input products after new input file has been merged
  bool
  Principal::adjustToNewProductRegistry(ProductRegistry const& reg) {
    if(reg.constProductList().size() > productHolders_.size()) {
      return false;
    }
    ProductRegistry::ProductList const& prodsList = reg.productList();
    for(auto const& prod : prodsList) {
      BranchDescription const& bd = prod.second;
      if(!bd.produced() && (bd.branchType() == branchType_)) {
        boost::shared_ptr<ConstBranchDescription> cbd(new ConstBranchDescription(bd));
        ProductHolderBase* phb = getExistingProduct(cbd->branchID());
        if(phb == nullptr || phb->branchDescription().branchName() != cbd->branchName()) {
            return false;
        }
        phb->resetBranchDescription(cbd);
      }
    }
    return true;
  }

  void
  Principal::addScheduledProduct(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<ProductHolderBase> phb(new ScheduledProductHolder(bd));
    addProductOrThrow(phb);
  }

  void
  Principal::addSourceProduct(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<ProductHolderBase> phb(new SourceProductHolder(bd));
    addProductOrThrow(phb);
  }

  void
  Principal::addInputProduct(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<ProductHolderBase> phb(new InputProductHolder(bd));
    addProductOrThrow(phb);
  }

  void
  Principal::addUnscheduledProduct(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<ProductHolderBase> phb(new UnscheduledProductHolder(bd));
    addProductOrThrow(phb);
  }

  void
  Principal::addAliasedProduct(boost::shared_ptr<ConstBranchDescription> bd) {
    ProductTransientIndex index = preg_->indexFrom(bd->originalBranchID());
    assert(index != ProductRegistry::kInvalidIndex);

    std::auto_ptr<ProductHolderBase> phb(new AliasProductHolder(bd, dynamic_cast<ProducedProductHolder&>(*productHolders_[index])));
    addProductOrThrow(phb);
  }

  // "Zero" the principal so it can be reused for another Event.
  void
  Principal::clearPrincipal() {
    processHistoryPtr_ = 0;
    processHistoryID_ = ProcessHistoryID();
    reader_ = 0;
    for(auto const& prod : *this) {
      prod->resetProductData();
    }
    productPtrs_.clear();
  }

  void
  Principal::deleteProduct(BranchID const& id) {
    ProductHolderBase* phb = getExistingProduct(id);
    assert(nullptr != phb);
    auto itFound = productPtrs_.find(phb->product().get());
    if(itFound != productPtrs_.end()) {
      productPtrs_.erase(itFound);
    } 
    phb->deleteProduct();
  }
  
  // Set the principal for the Event, Lumi, or Run.
  void
  Principal::fillPrincipal(ProcessHistoryID const& hist, DelayedReader* reader) {
    if(reader) {
      reader_ = reader;
    }

    ProcessHistory const* inputProcessHistory = 0;
    if (historyAppender_ && productRegistry().anyProductProduced()) {
      CachedHistory const& cachedHistory = 
        historyAppender_->appendToProcessHistory(hist,
                                                *processConfiguration_);
      processHistoryPtr_ = cachedHistory.processHistory();
      processHistoryID_ = cachedHistory.processHistoryID();
      inputProcessHistory = cachedHistory.inputProcessHistory();
    }
    else {
      if (hist.isValid()) {
        ProcessHistoryRegistry* registry = ProcessHistoryRegistry::instance();
        inputProcessHistory = registry->getMapped(hist);
        if (inputProcessHistory == 0) {
          throw Exception(errors::LogicError)
            << "Principal::fillPrincipal\n"
            << "Input ProcessHistory not found in registry\n"
            << "Contact a Framework developer\n";
        }
      } else {
        inputProcessHistory = &emptyProcessHistory_;
      }
      processHistoryID_ = hist;
      processHistoryPtr_ = inputProcessHistory;        
    }

    preg_->productLookup().reorderIfNecessary(branchType_, *inputProcessHistory,
                                         processConfiguration_->processName());
    preg_->elementLookup().reorderIfNecessary(branchType_, *inputProcessHistory,
                                         processConfiguration_->processName());
  }

  ProductHolderBase*
  Principal::getExistingProduct(BranchID const& branchID) {
    ProductTransientIndex index = preg_->indexFrom(branchID);
    assert(index != ProductRegistry::kInvalidIndex);
    SharedProductPtr ptr = productHolders_.at(index);
    return ptr.get();
  }

  ProductHolderBase*
  Principal::getExistingProduct(ProductHolderBase const& productHolder) {
    ProductHolderBase* phb = getExistingProduct(productHolder.branchDescription().branchID());
    assert(nullptr == phb || BranchKey(productHolder.branchDescription()) == BranchKey(phb->branchDescription()));
    return phb;
  }

  void
  Principal::addProduct_(std::auto_ptr<ProductHolderBase> productHolder) {
    ConstBranchDescription const& bd = productHolder->branchDescription();
    assert (!bd.className().empty());
    assert (!bd.friendlyClassName().empty());
    assert (!bd.moduleLabel().empty());
    assert (!bd.processName().empty());
    SharedProductPtr phb(productHolder);

    ProductTransientIndex index = preg_->indexFrom(bd.branchID());
    assert(index != ProductRegistry::kInvalidIndex);
    productHolders_[index] = phb;
  }

  void
  Principal::addProductOrThrow(std::auto_ptr<ProductHolderBase> productHolder) {
    ProductHolderBase const* phb = getExistingProduct(*productHolder);
    if(phb != nullptr) {
      ConstBranchDescription const& bd = productHolder->branchDescription();
      throw Exception(errors::InsertFailure, "AlreadyPresent")
          << "addProductOrThrow: Problem found while adding product, "
          << "product already exists for ("
          << bd.friendlyClassName() << ","
          << bd.moduleLabel() << ","
          << bd.productInstanceName() << ","
          << bd.processName()
          << ")\n";
    }
    addProduct_(productHolder);
  }

  Principal::ConstProductPtr
  Principal::getProductHolder(BranchID const& bid, bool resolveProd, bool fillOnDemand) const {
    ProductTransientIndex index = preg_->indexFrom(bid);
    if(index == ProductRegistry::kInvalidIndex){
       return ConstProductPtr();
    }
    return getProductByIndex(index, resolveProd, fillOnDemand);
  }

  Principal::ConstProductPtr
  Principal::getProductByIndex(ProductTransientIndex const& index, bool resolveProd, bool fillOnDemand) const {

    ConstProductPtr const phb = productHolders_[index].get();
    if(nullptr == phb) {
      return phb;
    }
    if(resolveProd && !phb->productUnavailable()) {
      this->resolveProduct(*phb, fillOnDemand);
    }
    return phb;
  }

  BasicHandle
  Principal::getByLabel(TypeID const& productType,
                        std::string const& label,
                        std::string const& productInstanceName,
                        std::string const& processName,
                        size_t& cachedOffset,
                        int& fillCount) const {

    ProductData const* result = findProductByLabel(productType,
                                                 preg_->productLookup(),
                                                 label,
                                                 productInstanceName,
                                                 processName,
                                                 cachedOffset,
                                                 fillCount);

    if(result == 0) {
      boost::shared_ptr<cms::Exception> whyFailed = makeNotFoundException("getByLabel", productType, label, productInstanceName, processName);
      return BasicHandle(whyFailed);
    }
    return BasicHandle(*result);
  }

  void
  Principal::getManyByType(TypeID const& productType,
                           BasicHandleVec& results) const {

    findProducts(productType,
               preg_->productLookup(),
               results);
    return;
  }

  size_t
  Principal::getMatchingSequence(TypeID const& typeID,
                                 std::string const& moduleLabel,
                                 std::string const& productInstanceName,
                                 std::string const& processName,
                                 BasicHandle& result) const {

    return findProduct(typeID,
                     preg_->elementLookup(),
                     moduleLabel,
                     productInstanceName,
                     processName,
                     result);
  }

  size_t
  Principal::findProducts(TypeID const& typeID,
                        TransientProductLookupMap const& typeLookup,
                        BasicHandleVec& results) const {
    assert(results.empty());

    typedef TransientProductLookupMap TypeLookup;
    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not, we throw an exception.
    // The missing dictionary might be for the class itself, the wrapped class, or a component of the class.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID, branchType_));
    if(range.first == range.second) {
      maybeThrowMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup(), preg_->missingDictionaries());
    }

    results.reserve(range.second - range.first);

    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {

      ConstBranchDescription const& bd = *(it->branchDescription());

      // Ignore aliases to avoid matching the same product multiple times.
      if(bd.isAlias()) {
        continue;
      }

      //now see if the data is actually available
      ConstProductPtr const& productHolder = getProductByIndex(it->index(), false, false);
      //NOTE sometimes 'productHolder->productUnavailable()' is true if was already deleted
      if(productHolder && productHolder->productWasDeleted()) {
        throwProductDeletedException("findProducts",
                                     typeID,
                                     bd.moduleLabel(),
                                     bd.productInstanceName(),
                                     bd.processName());
      }

      // Skip product if not available.
      if(productHolder && !productHolder->productUnavailable()) {

        this->resolveProduct(*productHolder, true);
        // If the product is a dummy filler, product holder will now be marked unavailable.
        // Unscheduled execution can fail to produce the EDProduct so check
        if(productHolder->product() && !productHolder->productUnavailable() && !productHolder->onDemand()) {
          // Found a good match, save it
          BasicHandle bh(productHolder->productData());
          results.push_back(bh);
        }
      }
    }
    return results.size();
  }

  size_t
  Principal::findProduct(TypeID const& typeID,
                       TransientProductLookupMap const& typeLookup,
                       std::string const& moduleLabel,
                       std::string const& productInstanceName,
                       std::string const& processName,
                       BasicHandle& result) const {
    assert(!result.isValid());

    size_t count = 0U;

    typedef TransientProductLookupMap TypeLookup;
    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not, we throw an exception.
    // The missing dictionary might be for the class itself, the wrapped class, or a component of the class.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID, branchType_));
    if(range.first == range.second) {
      maybeThrowMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup(), preg_->missingDictionaries());
    }

    unsigned int processLevelFound = std::numeric_limits<unsigned int>::max();
    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
      if(it->processIndex() > processLevelFound) {
        //this is for a less recent process and we've already found a match for a more recent process
        continue;
      }

      ConstBranchDescription const& bd = *(it->branchDescription());

      if ( moduleLabel == bd.moduleLabel() &&
           productInstanceName == bd.productInstanceName() &&
           (processName.empty() || processName == bd.processName())) {

        //now see if the data is actually available
        ConstProductPtr const& productHolder = getProductByIndex(it->index(), false, false);
        if(productHolder && productHolder->productWasDeleted()) {
          throwProductDeletedException("findProduct",
                                       typeID,
                                       bd.moduleLabel(),
                                       bd.productInstanceName(),
                                       bd.processName());
        }

        // Skip product if not available.
        if(productHolder && !productHolder->productUnavailable()) {

          this->resolveProduct(*productHolder, true);
          // If the product is a dummy filler, product holder will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if(productHolder->product() && !productHolder->productUnavailable() && !productHolder->onDemand()) {
            if(it->processIndex() < processLevelFound) {
              processLevelFound = it->processIndex();
              count = 0U;
            }
            if(count == 0U) {
              // Found a unique (so far) match, save it
              result = BasicHandle(productHolder->productData());
            }
            ++count;
          }
        }
      }
    }
    if(count != 1) result = BasicHandle();
    return count;
  }

  ProductData const*
  Principal::findProductByLabel(TypeID const& typeID,
                              TransientProductLookupMap const& typeLookup,
                              std::string const& moduleLabel,
                              std::string const& productInstanceName,
                              std::string const& processName,
                              size_t& cachedOffset,
                              int& fillCount) const {

    typedef TransientProductLookupMap TypeLookup;
    bool isCached = (fillCount > 0 && fillCount == typeLookup.fillCount());
    bool toBeCached = (fillCount >= 0 && !isCached);

    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> range =
        (isCached ? std::make_pair(typeLookup.begin() + cachedOffset, typeLookup.end()) : typeLookup.equal_range(TypeInBranchType(typeID, branchType_), moduleLabel, productInstanceName));

    if(toBeCached) {
      cachedOffset = range.first - typeLookup.begin();
      fillCount = typeLookup.fillCount();
    }

    if(range.first == range.second) {
      if(toBeCached) {
        cachedOffset = typeLookup.end() - typeLookup.begin();
      }
      // We check for a missing dictionary.  We do this only in this error leg.
      // A class without a dictionary cannot be in an Event/Lumi/Run.
      // We check if the class has a dictionary.  If it does not, we throw an exception.
      // The missing dictionary might be for the class itself, the wrapped class, or a component of the class.
      std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const typeRange = typeLookup.equal_range(TypeInBranchType(typeID, branchType_));
      if(typeRange.first == typeRange.second) {
        maybeThrowMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup(), preg_->missingDictionaries());
      }
      return 0;
    }

    if(!processName.empty()) {
      if(isCached) {
        assert(processName == range.first->branchDescription()->processName());
        range.second = range.first + 1;
      } else if(toBeCached) {
        bool processFound = false;
        for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
          if(it->isFirst() && it != range.first) {
            break;
          }
          if(processName == it->branchDescription()->processName()) {
            processFound = true;
            range.first = it;
            cachedOffset = range.first - typeLookup.begin();
            range.second = range.first + 1;
            break;
          }
        }
        if(!processFound) {
          cachedOffset = typeLookup.end() - typeLookup.begin();
          return 0;
        }
      } // end if(toBeCached)
    }

    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
      if(it->isFirst() && it != range.first) {
        return 0;
      }
      if(!processName.empty() && processName != it->branchDescription()->processName()) {
        continue;
      }
      //now see if the data is actually available
      ConstProductPtr const& productHolder = getProductByIndex(it->index(), false, false);
      if(productHolder && productHolder->productWasDeleted()) {
        throwProductDeletedException("findProductByLabel",
                                     typeID,
                                     moduleLabel,
                                     productInstanceName,
                                     processName);
      }

      // Skip product if not available.
      if(productHolder && !productHolder->productUnavailable()) {
        this->resolveProduct(*productHolder, true);
        // If the product is a dummy filler, product holder will now be marked unavailable.
        // Unscheduled execution can fail to produce the EDProduct so check
        if(productHolder->product() && !productHolder->productUnavailable() && !productHolder->onDemand()) {
          // Found the match
          return &productHolder->productData();
        }
      }
    }
    return 0;
  }

  ProductData const*
  Principal::findProductByTag(TypeID const& typeID, InputTag const& tag) const {
    ProductData const* productData =
        findProductByLabel(typeID,
                         preg_->productLookup(),
                         tag.label(),
                         tag.instance(),
                         tag.process(),
                         tag.cachedOffset(),
                         tag.fillCount());
    if(productData == 0) {
      throwNotFoundException("findProductByTag", typeID, tag);
    }
    return productData;
  }

  OutputHandle
  Principal::getForOutput(BranchID const& bid, bool getProd) const {
    ConstProductPtr const phb = getProductHolder(bid, getProd, true);
    if(phb == nullptr) {
      throwProductNotFoundException("getForOutput", errors::LogicError, bid);
    }
    if (phb->productWasDeleted()) {
      throwProductDeletedException("getForOutput",phb->productType(),
                                   phb->moduleLabel(),
                                   phb->productInstanceName(),
                                   phb->processName());
    }
    if(!phb->provenance() || (!phb->product() && !phb->productProvenancePtr())) {
      return OutputHandle();
    }
    return OutputHandle(WrapperHolder(phb->product().get(), phb->productData().getInterface()), &phb->branchDescription(), phb->productProvenancePtr());
  }

  Provenance
  Principal::getProvenance(BranchID const& bid) const {
    ConstProductPtr const phb = getProductHolder(bid, false, true);
    if(phb == nullptr) {
      throwProductNotFoundException("getProvenance", errors::ProductNotFound, bid);
    }

    if(phb->onDemand()) {
      unscheduledFill(phb->branchDescription().moduleLabel());
    }
    // We already tried to produce the unscheduled products above
    // If they still are not there, then throw
    if(phb->onDemand()) {
      throwProductNotFoundException("getProvenance(onDemand)", errors::ProductNotFound, bid);
    }

    return *phb->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  Principal::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provenances.clear();
    for(auto const& productHolder : *this) {
      if(productHolder->provenanceAvailable() && !productHolder->branchDescription().isAlias()) {
        // We do not attempt to get the event/lumi/run status from the provenance,
        // because the per event provenance may have been dropped.
        if(productHolder->provenance()->product().present()) {
           provenances.push_back(productHolder->provenance());
        }
      }
    }
  }

  void
  Principal::recombine(Principal& other, std::vector<BranchID> const& bids) {
    for(auto const& prod : bids) {
      ProductTransientIndex index= preg_->indexFrom(prod);
      assert(index!=ProductRegistry::kInvalidIndex);
      ProductTransientIndex indexO = other.preg_->indexFrom(prod);
      assert(indexO!=ProductRegistry::kInvalidIndex);
      productHolders_[index].swap(other.productHolders_[indexO]);
    }
    reader_->mergeReaders(other.reader());
  }

  WrapperHolder
  Principal::getIt(ProductID const&) const {
    assert(nullptr);
    return WrapperHolder();
  }

  void
  Principal::maybeFlushCache(TypeID const& tid, InputTag const& tag) const {
    if(tag.typeID() != tid ||
        tag.branchType() != branchType() ||
        tag.productRegistry() != &productRegistry()) {
      tag.fillCount() = 0;
      tag.cachedOffset() = 0U;
      tag.typeID() = tid;
      tag.branchType() = branchType();
      tag.productRegistry() = &productRegistry();
    }
  }

  void
  Principal::checkUniquenessAndType(WrapperOwningHolder const& prod, ProductHolderBase const* phb) const {
    if(!prod.isValid()) return;
    // These are defensive checks against things that should never happen, but have.
    // Checks that the same physical product has not already been put into the event.
    bool alreadyPresent = !productPtrs_.insert(prod.wrapper()).second;
    if(alreadyPresent) {
      phb->checkType(prod);
      const_cast<WrapperOwningHolder&>(prod).reset();
      throwCorruptionException("checkUniquenessAndType", phb->branchDescription().branchName());
    }
    // Checks that the real type of the product matches the branch.
    phb->checkType(prod);
  }

  void
  Principal::putOrMerge(WrapperOwningHolder const& prod, ProductHolderBase const* phb) const {
    bool willBePut = phb->putOrMergeProduct();
    if(willBePut) {
      checkUniquenessAndType(prod, phb);
      phb->putProduct(prod);
    } else {
      phb->checkType(prod);
      phb->mergeProduct(prod);
    }
  }

  void
  Principal::putOrMerge(WrapperOwningHolder const& prod, ProductProvenance& prov, ProductHolderBase* phb) {
    bool willBePut = phb->putOrMergeProduct();
    if(willBePut) {
      checkUniquenessAndType(prod, phb);
      phb->putProduct(prod, prov);
    } else {
      phb->checkType(prod);
      phb->mergeProduct(prod, prov);
    }
  }

  void
  Principal::adjustIndexesAfterProductRegistryAddition() {
    if(preg_->constProductList().size() > productHolders_.size()) {
      ProductHolderCollection newProductHolders(preg_->constProductList().size(), SharedProductPtr());
      for(auto const& prod : *this) {
        ProductTransientIndex index = preg_->indexFrom(prod->branchDescription().branchID());
        assert(index != ProductRegistry::kInvalidIndex);
        newProductHolders[index] = prod;
      }
      productHolders_.swap(newProductHolders);
      // Now we must add new product holders for any new product registry entries.
      for(auto const& prod : preg_->productList()) {
        BranchDescription const& bd = prod.second;
        if(bd.branchType() == branchType_) {
          ProductTransientIndex index = preg_->indexFrom(bd.branchID());
          assert(index != ProductRegistry::kInvalidIndex);
          if(!productHolders_[index]) {
            // no product holder.  Must add one. The new entry must be an input product holder.
            assert(!bd.produced());
            boost::shared_ptr<ConstBranchDescription> cbd(new ConstBranchDescription(bd));
            addInputProduct(cbd);
          }
        }
      }
    }
    assert(preg_->constProductList().size() == productHolders_.size());
  }
}
