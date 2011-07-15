/**----------------------------------------------------------------------
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Principal.h"

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

//using boost::lambda::_1;

namespace edm {
  static
  void
  throwMultiFoundException(char const* where, int nFound, TypeID const& productType) {
    throw Exception(errors::ProductNotFound)
      << "Principal::" << where << ": Found " << nFound << " products rather than one which match all criteria\n"
      << "Looking for type: " << productType << "\n";
  }

  static
  void
  throwGroupNotFoundException(char const* where, errors::ErrorCodes error, BranchID const& bid) {
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
  makeNotFoundException(char const* where, TypeID const& productType) {
    boost::shared_ptr<cms::Exception> exception(new Exception(errors::ProductNotFound));
    *exception << "Principal::" << where << ": Found zero products matching all criteria\nLooking for type: " << productType << "\n";
    return exception;
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
  throwNotFoundException(char const* where, TypeID const& productType, InputTag const& tag) {
    boost::shared_ptr<cms::Exception> exception = makeNotFoundException(where, productType, tag.label(), tag.instance(), tag.process());
    throw *exception;
  }

  static
  void
  throwMissingDictionaryException(TypeID const& productType, bool isElement) {
    checkDictionaries(isElement ? productType.className() : wrappedClassName(productType.className()), false);
    throwMissingDictionariesException();
  }

  Principal::Principal(boost::shared_ptr<ProductRegistry const> reg,
                       ProcessConfiguration const& pc,
                       BranchType bt) :
    EDProductGetter(),
    processHistoryPtr_(new ProcessHistory),
    processHistoryID_(processHistoryPtr_->id()),
    processConfiguration_(&pc),
    groups_(reg->constProductList().size(), SharedGroupPtr()),
    preg_(reg),
    branchMapperPtr_(new BranchMapper),
    reader_(),
    productPtrs_(),
    branchType_(bt) {
    //Now that these have been set, we can create the list of Branches we need.
    std::string const source("source");
    ProductRegistry::ProductList const& prodsList = reg->productList();
    for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
          itProdInfoEnd = prodsList.end();
        itProdInfo != itProdInfoEnd;
        ++itProdInfo) {
      if(itProdInfo->second.branchType() == branchType_) {
        boost::shared_ptr<ConstBranchDescription> bd(new ConstBranchDescription(itProdInfo->second));
        if(bd->produced()) {
          if(bd->moduleLabel() == source) {
            addGroupSource(bd);
          } else if(bd->onDemand()) {
            assert(bt == InEvent);
            addOnDemandGroup(bd);
          } else {
            addGroupScheduled(bd);
          }
        } else {
          addGroupInput(bd);
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
    for(const_iterator it = this->begin(), itEnd = this->end(); it != itEnd; ++it) {
      Group const& g = **it;
      if(!g.productUnavailable() && !g.onDemand() && !g.branchDescription().dropped()) {
        ++size;
      }
    }
    return size;
  }

  //adjust provenance for input groups after new input file has been merged
  bool
  Principal::adjustToNewProductRegistry(ProductRegistry const& reg) {
    if(reg.constProductList().size() > groups_.size()) {
      return false;
    }
    ProductRegistry::ProductList const& prodsList = reg.productList();
    for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
          itProdInfoEnd = prodsList.end();
        itProdInfo != itProdInfoEnd;
        ++itProdInfo) {
    if(!itProdInfo->second.produced() && (itProdInfo->second.branchType() == branchType_)) {
        boost::shared_ptr<ConstBranchDescription> bd(new ConstBranchDescription(itProdInfo->second));
        Group *g = getExistingGroup(itProdInfo->second.branchID());
        if(g == 0 || g->branchDescription().branchName() != bd->branchName()) {
            return false;
        }
        g->resetBranchDescription(bd);
      }
    }
    return true;
  }

  void
  Principal::addGroupScheduled(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<Group> g(new ScheduledGroup(bd));
    addGroupOrThrow(g);
  }

  void
  Principal::addGroupSource(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<Group> g(new SourceGroup(bd));
    addGroupOrThrow(g);
  }

  void
  Principal::addGroupInput(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<Group> g(new InputGroup(bd));
    addGroupOrThrow(g);
  }

  void
  Principal::addOnDemandGroup(boost::shared_ptr<ConstBranchDescription> bd) {
    std::auto_ptr<Group> g(new UnscheduledGroup(bd));
    addGroupOrThrow(g);
  }

  // "Zero" the principal so it can be reused for another Event.
  void
  Principal::clearPrincipal() {
    processHistoryPtr_->clear();
    processHistoryID_ = processHistoryPtr_->id();
    branchMapperPtr_->reset();
    reader_ = 0;
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      (*i)->resetProductData();
    }
    productPtrs_.clear();
  }

  // Set the principal for the Event, Lumi, or Run.
  void
  Principal::fillPrincipal(ProcessHistoryID const& hist, boost::shared_ptr<BranchMapper> mapper, DelayedReader* reader) {
    if(mapper) {
      branchMapperPtr_ = mapper;
    }
    if(reader) {
      reader_ = reader;
    }
    if(hist.isValid()) {
      ProcessHistoryRegistry& history(*ProcessHistoryRegistry::instance());
      assert(history.notEmpty());
      bool found = history.getMapped(hist, *processHistoryPtr_);
      assert(found);
      processHistoryID_ = processHistoryPtr_->id();
    }
    preg_->productLookup().reorderIfNecessary(branchType_, *processHistoryPtr_,
                                         processConfiguration_->processName());
    preg_->elementLookup().reorderIfNecessary(branchType_, *processHistoryPtr_,
                                         processConfiguration_->processName());
  }

  Group*
  Principal::getExistingGroup(BranchID const& branchID) {
    ProductTransientIndex index = preg_->indexFrom(branchID);
    assert(index != ProductRegistry::kInvalidIndex);
    SharedGroupPtr ptr = groups_.at(index);
    return ptr.get();
  }

  Group*
  Principal::getExistingGroup(Group const& group) {
    Group* g = getExistingGroup(group.branchDescription().branchID());
    assert(0 == g || BranchKey(group.branchDescription()) == BranchKey(g->branchDescription()));
    return g;
  }

  void
  Principal::addGroup_(std::auto_ptr<Group> group) {
    ConstBranchDescription const& bd = group->branchDescription();
    assert (!bd.className().empty());
    assert (!bd.friendlyClassName().empty());
    assert (!bd.moduleLabel().empty());
    assert (!bd.processName().empty());
    SharedGroupPtr g(group);

    ProductTransientIndex index = preg_->indexFrom(bd.branchID());
    assert(index != ProductRegistry::kInvalidIndex);
    groups_[index] = g;
  }

  void
  Principal::addGroupOrThrow(std::auto_ptr<Group> group) {
    Group const* g = getExistingGroup(*group);
    if(g != 0) {
      ConstBranchDescription const& bd = group->branchDescription();
      throw Exception(errors::InsertFailure, "AlreadyPresent")
          << "addGroupOrThrow: Problem found while adding product, "
          << "product already exists for ("
          << bd.friendlyClassName() << ","
          << bd.moduleLabel() << ","
          << bd.productInstanceName() << ","
          << bd.processName()
          << ")\n";
    }
    addGroup_(group);
  }

  void
  Principal::setProcessHistory(Principal const& principal) {
    *processHistoryPtr_ = *principal.processHistoryPtr_;
    processHistoryID_ = processHistoryPtr_->id();
  }

  Principal::ConstGroupPtr const
  Principal::getGroup(BranchID const& bid, bool resolveProd, bool fillOnDemand) const {
    ProductTransientIndex index = preg_->indexFrom(bid);
    if(index == ProductRegistry::kInvalidIndex){
       return ConstGroupPtr();
    }
    return getGroupByIndex(index, resolveProd, fillOnDemand);
  }

  Principal::ConstGroupPtr const
  Principal::getGroupByIndex(ProductTransientIndex const& index, bool resolveProd, bool fillOnDemand) const {

    ConstGroupPtr const g = groups_[index].get();
    if(0 == g) {
      return g;
    }
    if(resolveProd && !g->productUnavailable()) {
      this->resolveProduct(*g, fillOnDemand);
    }
    return g;
  }

  BasicHandle
  Principal::getBySelector(TypeID const& productType,
                           SelectorBase const& sel) const {
    BasicHandle result;

    int nFound = findGroup(productType,
                           preg_->productLookup(),
                           sel,
                           result);

    if(nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed = makeNotFoundException("getBySelector", productType);
      return BasicHandle(whyFailed);
    }
    if(nFound > 1) {
      throwMultiFoundException("getBySelector", nFound, productType);
    }
    return result;
  }

  BasicHandle
  Principal::getByLabel(TypeID const& productType,
                        std::string const& label,
                        std::string const& productInstanceName,
                        std::string const& processName,
                        size_t& cachedOffset,
                        int& fillCount) const {

    ProductData const* result = findGroupByLabel(productType,
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
  Principal::getMany(TypeID const& productType,
                     SelectorBase const& sel,
                     BasicHandleVec& results) const {

    findGroups(productType,
               preg_->productLookup(),
               sel,
               results);

    return;
  }

  BasicHandle
  Principal::getByType(TypeID const& productType) const {

    BasicHandle result;
    MatchAllSelector sel;

    int nFound = findGroup(productType,
                           preg_->productLookup(),
                           sel,
                           result);

    if(nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed = makeNotFoundException("getByType", productType);
      return BasicHandle(whyFailed);
    }

    if(nFound > 1) {
      throwMultiFoundException("getByType", nFound, productType);
    }
    return result;
  }

  void
  Principal::getManyByType(TypeID const& productType,
                           BasicHandleVec& results) const {

    MatchAllSelector sel;

    findGroups(productType,
               preg_->productLookup(),
               sel,
               results);
    return;
  }

  size_t
  Principal::getMatchingSequence(TypeID const& typeID,
                                 SelectorBase const& selector,
                                 BasicHandle& result) const {

    // One new argument is the element lookup container
    // Otherwise this just passes through the arguments to findGroup
    return findGroup(typeID,
                     preg_->elementLookup(),
                     selector,
                     result);
  }

  size_t
  Principal::findGroups(TypeID const& typeID,
                        TransientProductLookupMap const& typeLookup,
                        SelectorBase const& selector,
                        BasicHandleVec& results) const {
    assert(results.empty());

    typedef TransientProductLookupMap TypeLookup;
    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not, we throw an exception.
    // The missing dictionary might be for the class itself, the wrapped class, or a component of the class.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID, branchType_));
    if(range.first == range.second) {
      throwMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup());
    }

    results.reserve(range.second - range.first);

    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {

      if(selector.match(*(it->branchDescription()))) {

        //now see if the data is actually available
        ConstGroupPtr const& group = getGroupByIndex(it->index(), false, false);
        // Skip product if not available.
        if(group && !group->productUnavailable()) {
          this->resolveProduct(*group, true);
          // If the product is a dummy filler, group will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if(!group->productUnavailable() && !group->onDemand()) {
            // Found a good match, save it
            BasicHandle bh(group->productData());
            results.push_back(bh);
          }
        }
      }
    }
    return results.size();
  }

  size_t
  Principal::findGroup(TypeID const& typeID,
                       TransientProductLookupMap const& typeLookup,
                       SelectorBase const& selector,
                       BasicHandle& result) const {
    assert(!result.isValid());

    size_t count = 0U;

    typedef TransientProductLookupMap TypeLookup;
    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not, we throw an exception.
    // The missing dictionary might be for the class itself, the wrapped class, or a component of the class.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID, branchType_));
    if(range.first == range.second) {
      throwMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup());
    }

    unsigned int processLevelFound = std::numeric_limits<unsigned int>::max();
    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
      if(it->processIndex() > processLevelFound) {
        //this is for a less recent process and we've already found a match for a more recent process
        continue;
      }

      if(selector.match(*(it->branchDescription()))) {

        //now see if the data is actually available
        ConstGroupPtr const& group = getGroupByIndex(it->index(), false, false);
        // Skip product if not available.
        if(group && !group->productUnavailable()) {
          this->resolveProduct(*group, true);
          // If the product is a dummy filler, group will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if(!group->productUnavailable() && !group->onDemand()) {
            if(it->processIndex() < processLevelFound) {
              processLevelFound = it->processIndex();
              count = 0U;
            }
            if(count == 0U) {
              // Found a unique (so far) match, save it
              result = BasicHandle(group->productData());
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
  Principal::findGroupByLabel(TypeID const& typeID,
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
        throwMissingDictionaryException(typeID, &typeLookup == &preg_->elementLookup());
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
      ConstGroupPtr const& group = getGroupByIndex(it->index(), false, false);
      // Skip product if not available.
      if(group && !group->productUnavailable()) {
        this->resolveProduct(*group, true);
        // If the product is a dummy filler, group will now be marked unavailable.
        // Unscheduled execution can fail to produce the EDProduct so check
        if(!group->productUnavailable() && !group->onDemand()) {
          // Found the match
          return &group->productData();
        }
      }
    }
    return 0;
  }

  ProductData const*
  Principal::findGroupByTag(TypeID const& typeID, InputTag const& tag) const {
    ProductData const* productData =
        findGroupByLabel(typeID,
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
    ConstGroupPtr const g = getGroup(bid, getProd, false);
    if(g == 0) {
      throwGroupNotFoundException("getForOutput", errors::LogicError, bid);
    }
    if(!g->provenance() || (!g->product() && !g->productProvenancePtr())) {
      return OutputHandle();
    }
    return OutputHandle(WrapperHolder(g->product().get(), g->productData().getInterface()), &g->branchDescription(), g->productProvenancePtr());
  }

  Provenance
  Principal::getProvenance(BranchID const& bid) const {
    ConstGroupPtr const g = getGroup(bid, false, true);
    if(g == 0) {
      throwGroupNotFoundException("getProvenance", errors::ProductNotFound, bid);
    }

    if(g->onDemand()) {
      unscheduledFill(g->branchDescription().moduleLabel());
    }
    // We already tried to produce the unscheduled products above
    // If they still are not there, then throw
    if(g->onDemand()) {
      throwGroupNotFoundException("getProvenance(onDemand)", errors::ProductNotFound, bid);
    }

    return *g->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  Principal::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provenances.clear();
    for (const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      if((*i)->provenanceAvailable()) {
        // We do not attempt to get the event/lumi/run status from the provenance,
        // because the per event provenance may have been dropped.
        if((*i)->provenance()->product().present()) {
           provenances.push_back((*i)->provenance());
        }
      }
    }
  }

  void
  Principal::recombine(Principal& other, std::vector<BranchID> const& bids) {
    for (std::vector<BranchID>::const_iterator it = bids.begin(), itEnd = bids.end(); it != itEnd; ++it) {
      ProductTransientIndex index= preg_->indexFrom(*it);
      assert(index!=ProductRegistry::kInvalidIndex);
      ProductTransientIndex indexO = other.preg_->indexFrom(*it);
      assert(indexO!=ProductRegistry::kInvalidIndex);
      groups_[index].swap(other.groups_[indexO]);
    }
    reader_->mergeReaders(other.reader());
    branchMapperPtr_->mergeMappers(other.branchMapperPtr());
  }

  WrapperHolder
  Principal::getIt(ProductID const&) const {
    assert(0);
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
  Principal::checkUniquenessAndType(WrapperOwningHolder const& prod, Group const* g) const {
    if(!prod.isValid()) return;
    // These are defensive checks against things that should never happen, but have.
    // Checks that the same physical product has not already been put into the event.
    bool alreadyPresent = !productPtrs_.insert(prod.wrapper()).second;
    if(alreadyPresent) {
      g->checkType(prod);
      const_cast<WrapperOwningHolder&>(prod).reset();
      throwCorruptionException("checkUniquenessAndType", g->branchDescription().branchName());
    }
    // Checks that the real type of the product matches the branch.
    g->checkType(prod);
  }

  void
  Principal::putOrMerge(WrapperOwningHolder const& prod, Group const* g) const {
    bool willBePut = g->putOrMergeProduct();
    if(willBePut) {
      checkUniquenessAndType(prod, g);
      g->putProduct(prod);
    } else {
      g->checkType(prod);
      g->mergeProduct(prod);
    }
  }

  void
  Principal::putOrMerge(WrapperOwningHolder const& prod, ProductProvenance& prov, Group* g) {
    bool willBePut = g->putOrMergeProduct();
    if(willBePut) {
      checkUniquenessAndType(prod, g);
      g->putProduct(prod, prov);
    } else {
      g->checkType(prod);
      g->mergeProduct(prod, prov);
    }
  }

  void
  Principal::adjustIndexesAfterProductRegistryAddition() {
    if(preg_->constProductList().size() > groups_.size()) {
      GroupCollection newGroups(preg_->constProductList().size(), SharedGroupPtr());
      for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
        ProductTransientIndex index = preg_->indexFrom((*i)->branchDescription().branchID());
        assert(index != ProductRegistry::kInvalidIndex);
        newGroups[index] = *i;
      }
      groups_.swap(newGroups);
      // Now we must add new groups for any new product registry entries.
      ProductRegistry::ProductList const& prodsList = preg_->productList();
      for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
          itProdInfoEnd = prodsList.end();
          itProdInfo != itProdInfoEnd;
          ++itProdInfo) {
        if(itProdInfo->second.branchType() == branchType_) {
          ProductTransientIndex index = preg_->indexFrom(itProdInfo->second.branchID());
          assert(index != ProductRegistry::kInvalidIndex);
          if(!groups_[index]) {
            // no group.  Must add one. The new entry must be an input group.
            assert(!itProdInfo->second.produced());
            boost::shared_ptr<ConstBranchDescription> bd(new ConstBranchDescription(itProdInfo->second));
            addGroupInput(bd);
          }
        }
      }
    }
  }
}
