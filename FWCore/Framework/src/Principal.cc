/**----------------------------------------------------------------------
    clearPrincipal();
  ----------------------------------------------------------------------*/

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <cstring>

#include "FWCore/Framework/interface/Principal.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/Selector.h"
//using boost::lambda::_1;

namespace edm {

  Principal::Principal(boost::shared_ptr<ProductRegistry const> reg,
                       ProcessConfiguration const& pc,
                       BranchType bt) :
    EDProductGetter(),
    processHistoryPtr_(boost::shared_ptr<ProcessHistory>(new ProcessHistory)),
    processConfiguration_(&pc),
    processHistoryModified_(false),
    groups_(reg->constProductList().size(), SharedGroupPtr()),
    preg_(reg),
    branchMapperPtr_(),
    store_(),
    branchType_(bt) {
    //Now that these have been set, we can create the list of Branches we need.
    std::string const source("source");
    ProductRegistry::ProductList const& prodsList = reg->productList();
    for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
          itProdInfoEnd = prodsList.end();
        itProdInfo != itProdInfoEnd;
        ++itProdInfo) {
      if (itProdInfo->second.branchType() == branchType_) {
        boost::shared_ptr<ConstBranchDescription> bd(new ConstBranchDescription(itProdInfo->second));
        if (bd->produced()) {
          if (bd->moduleLabel() == source) {
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
      if (!g.productUnavailable() && !g.onDemand() && !g.branchDescription().dropped()) {
	++size;
      }
    }
    return size;
  }

  //Reset provenance for input groups after new input file has been merged
  void
  Principal::reinitializeGroups(boost::shared_ptr<ProductRegistry const> reg) {
    ProductRegistry::ProductList const& prodsList = reg->productList();
    for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
          itProdInfoEnd = prodsList.end();
        itProdInfo != itProdInfoEnd;
        ++itProdInfo) {
      if (!itProdInfo->second.produced() && (itProdInfo->second.branchType() == branchType_)) {
        boost::shared_ptr<ConstBranchDescription> bd(new ConstBranchDescription(itProdInfo->second));
        Group *g = getExistingGroup(itProdInfo->second.branchID());
        if(g != 0) {
	   g->resetBranchDescription(bd);
        } else {
          addGroupInput(bd);
        }
      }
    }
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

  void
  Principal::clearPrincipal() {
    processHistoryModified_ = false;
    processHistoryPtr_.reset(new ProcessHistory);
    branchMapperPtr_.reset();
    store_.reset();
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      (*i)->resetGroupData();
    }
  }

  void
  Principal::fillPrincipal(ProcessHistoryID const& hist, boost::shared_ptr<BranchMapper> mapper, boost::shared_ptr<DelayedReader> rtrv) {
    branchMapperPtr_ = mapper;
    store_ = rtrv;
    if (hist.isValid()) {
      ProcessHistoryRegistry& history(*ProcessHistoryRegistry::instance());
      assert(history.notEmpty());
      bool found = history.getMapped(hist, *processHistoryPtr_);
      assert(found);
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
    SharedGroupPtr ptr = groups_[index];
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
    if (g != 0) {
      ConstBranchDescription const& bd = group->branchDescription();
      throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
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
  Principal::addToProcessHistory() const {
    if (processHistoryModified_) return;
    ProcessHistory& ph = *processHistoryPtr_;
    std::string const& processName = processConfiguration_->processName();
    for (ProcessHistory::const_iterator it = ph.begin(), itEnd = ph.end(); it != itEnd; ++it) {
      if (processName == it->processName()) {
	throw edm::Exception(errors::Configuration, "Duplicate Process")
	  << "The process name " << processName << " was previously used on these products.\n"
	  << "Please modify the configuration file to use a distinct process name.\n";
      }
    }
    ph.push_back(*processConfiguration_);
    //OPTIMIZATION NOTE:  As of 0_9_0_pre3
    // For very simple Sources (e.g. EmptySource) this routine takes up nearly 50% of the time per event.
    // 96% of the time for this routine is being spent in computing the
    // ProcessHistory id which happens because we are reconstructing the ProcessHistory for each event.
    // (The process ID is first computed in the call to 'insertMapped(..)' below.)
    // It would probably be better to move the ProcessHistory construction out to somewhere
    // which persists for longer than one Event
    ProcessHistoryRegistry::instance()->insertMapped(ph);
    setProcessHistoryID(ph.id());
    processHistoryModified_ = true;
  }

  ProcessHistory const&
  Principal::processHistory() const {
    return *processHistoryPtr_;
  }

  Principal::SharedConstGroupPtr const
  Principal::getGroup(BranchID const& bid, bool resolveProd, bool resolveProv, bool fillOnDemand) const {
    ProductTransientIndex index = preg_->indexFrom(bid);
    if(index == ProductRegistry::kInvalidIndex){
       return SharedConstGroupPtr();
    }
    return getGroupByIndex(index, resolveProd, resolveProv, fillOnDemand); 
  }
   
  Principal::SharedConstGroupPtr const
  Principal::getGroupByIndex(ProductTransientIndex const& index, bool resolveProd, bool resolveProv, bool fillOnDemand) const {
    
    SharedConstGroupPtr const& g = groups_[index];
    if (0 == g.get()) {
      return g;
    }
    if (resolveProv && (g->provenanceAvailable() || g->onDemand())) {
      if(g->onDemand()) {
         //must execute the unscheduled to get the provenance
         this->resolveProduct(*g, true);
         //check if this failed (say because of a caught exception)
         if(0 == g->product()) {
            return g;
         }
      }
      this->resolveProvenance(*g);
    }
    if (resolveProd && !g->productUnavailable()) {
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

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
      *whyFailed
	<< "getBySelector: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
      return BasicHandle(whyFailed);
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getBySelector: Found " << nFound << " products rather than one which match all criteria\n"
	<< "Looking for type: " << productType << "\n";
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

    BasicHandle result;
    
    bool found = findGroupByLabel(productType,
                                  preg_->productLookup(),
                                  label,
                                  productInstanceName,
                                  processName,
			          cachedOffset,
			          fillCount,
                                  result);

    if (!found) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
      *whyFailed
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n"
	<< (processName.empty() ? "" : "Looking for process: ") << processName << "\n";
      return BasicHandle(whyFailed);
    }
    return result;
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

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
      *whyFailed
	<< "getByType: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
      return BasicHandle(whyFailed);
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByType: Found " << nFound << " products rather than one which match all criteria\n"
	<< "Looking for type: " << productType << "\n";
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
    // First, we check if the class has a dictionary.  If it does not,
    // we return immediately.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID,branchType_));
    if(range.first == range.second) {
      return 0;
    }

    results.reserve(range.second - range.first);
    
    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
        
      if(selector.match(*(it->branchDescription()))) {
        
        //now see if the data is actually available
        SharedConstGroupPtr const& group = getGroupByIndex(it->index(), false, false, false);
        // Skip product if not available.
        if (group && !group->productUnavailable()) {
          this->resolveProduct(*group, true);
	  // If the product is a dummy filler, group will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if (!group->productUnavailable() && !group->onDemand()) {
            // Found a good match, save it
	    BasicHandle bh(group->product(), group->provenance());
	    bh.provenance()->setStore(branchMapperPtr_);
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
    // First, we check if the class has a dictionary.  If it does not,
    // we return immediately.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID,branchType_));
    if(range.first == range.second) {
      return 0;
    }

    unsigned int processLevelFound = std::numeric_limits<unsigned int>::max();
    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
      if(it->processIndex() > processLevelFound) {
        //this is for a less recent process and we've already found a match for a more recent process
        continue;
      }
        
      if(selector.match(*(it->branchDescription()))) {
        
        //now see if the data is actually available
        SharedConstGroupPtr const& group = getGroupByIndex(it->index(), false, false, false);
        // Skip product if not available.
        if (group && !group->productUnavailable()) {
          this->resolveProduct(*group, true);
	  // If the product is a dummy filler, group will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if (!group->productUnavailable() && !group->onDemand()) {
            if(it->processIndex() < processLevelFound) {
              processLevelFound = it->processIndex();
	      count = 0U;
            }
	    if (count == 0U) {
              // Found a unique (so far) match, save it
	      result = BasicHandle(group->product(), group->provenance());
	      result.provenance()->setStore(branchMapperPtr_);
	    }
	    ++count;
          }
        }
      }
    }
    if (count != 1) result = BasicHandle();
    return count;
  }

  bool
  Principal::findGroupByLabel(TypeID const& typeID,
			      TransientProductLookupMap const& typeLookup,
			      std::string const& moduleLabel,
			      std::string const& productInstanceName,
			      std::string const& processName,
			      size_t& cachedOffset,
			      int& fillCount,
			      BasicHandle& result) const {
    assert(!result.isValid());

    typedef TransientProductLookupMap TypeLookup;
    bool isCached = (fillCount > 0  && fillCount == typeLookup.fillCount());
    bool toBeCached = (fillCount >= 0 && !isCached);

    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> range =
        (isCached ? std::make_pair(typeLookup.begin() + cachedOffset, typeLookup.end()) : typeLookup.equal_range(TypeInBranchType(typeID,branchType_), moduleLabel, productInstanceName));

    if (toBeCached) {
      cachedOffset = range.first - typeLookup.begin();
      fillCount = typeLookup.fillCount();
    }

    if(range.first == range.second) {
      if (toBeCached) {
        cachedOffset = typeLookup.end() - typeLookup.begin();
      }
      return false;
    }

    if (!processName.empty()) {
      if (isCached) {
        assert(processName == range.first->branchDescription()->processName());
        range.second = range.first + 1;
      } else if (toBeCached) {
	bool processFound = false;
	for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
          if (it->isFirst() && it != range.first) {
	    break;
          }
	  if (processName == it->branchDescription()->processName()) {
            processFound = true;
	    range.first = it;
            cachedOffset = range.first - typeLookup.begin();
            range.second = range.first + 1;
	    break;
	  }
	}
	if (!processFound) {
          cachedOffset = typeLookup.end() - typeLookup.begin();
	  return false;
	}
      } // end if(toBeCached)
    }

    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
      if (it->isFirst() && it != range.first) {
	return false;
      }
      if (!processName.empty() && processName != it->branchDescription()->processName()) {
        continue;
      }
      //now see if the data is actually available
      SharedConstGroupPtr const& group = getGroupByIndex(it->index(), false, false, false);
      // Skip product if not available.
      if (group && !group->productUnavailable()) {
        this->resolveProduct(*group, true);
	// If the product is a dummy filler, group will now be marked unavailable.
        // Unscheduled execution can fail to produce the EDProduct so check
        if (!group->productUnavailable() && !group->onDemand()) {
          // Found the match
	  result = BasicHandle(group->product(), group->provenance());
	  result.provenance()->setStore(branchMapperPtr_);
	  return true;
        }
      }
    }
    return false;
  }

  OutputHandle
  Principal::getForOutput(BranchID const& bid, bool getProd) const {
    SharedConstGroupPtr const& g = getGroup(bid, getProd, true, false);
    if (g.get() == 0) {
        throw edm::Exception(edm::errors::LogicError, "Principal::getForOutput\n")
         << "No entry is present for this branch.\n"
         << "The branch id is " << bid << "\n"
         << "Contact a framework developer.\n";
    }
    if (getProd && (g->product() == 0 || !g->product()->isPresent()) &&
	    g->branchDescription().present() &&
	    g->branchDescription().branchType() == InEvent &&
	    g->productProvenancePtr() &&
            productstatus::present(g->productProvenancePtr()->productStatus())) {
        throw edm::Exception(edm::errors::LogicError, "Principal::getForOutput\n")
         << "A product with a status of 'present' is not actually present.\n"
         << "The branch name is " << g->branchDescription().branchName() << "\n"
         << "Contact a framework developer.\n";
    }
    if (!g->product() && !g->productProvenancePtr()) {
      return OutputHandle();
    }
    return OutputHandle(g->product().get(), &g->branchDescription(), g->productProvenancePtr());
  }

  Provenance
  Principal::getProvenance(BranchID const& bid) const {
    SharedConstGroupPtr const& g = getGroup(bid, false, true, true);
    if (g.get() == 0) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given branch id: "<< bid << "\n";
    }

    if (g->onDemand()) {
      unscheduledFill(g->branchDescription().moduleLabel());
    }
    // We already tried to produce the unscheduled products above
    // If they still are not there, then throw
    if (g->onDemand()) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getProvenance: no product with given BranchID: "<< bid <<"\n";
    }

    return *g->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  Principal::getAllProvenance(std::vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      if ((*i)->provenanceAvailable()) {
        resolveProvenance(**i);
        if ((*i)->provenance()->productProvenanceSharedPtr() &&
            (*i)->provenance()->isPresent() &&
            (*i)->provenance()->product().present())
           provenances.push_back((*i)->provenance());
        }
    }
  }

  void
  Principal::recombine(Principal & other, std::vector<BranchID> const& bids) {
    for (std::vector<BranchID>::const_iterator it = bids.begin(), itEnd = bids.end(); it != itEnd; ++it) {
      ProductTransientIndex index= preg_->indexFrom(*it);
      assert(index!=ProductRegistry::kInvalidIndex);
      ProductTransientIndex indexO = other.preg_->indexFrom(*it);
      assert(indexO!=ProductRegistry::kInvalidIndex);
      groups_[index].swap(other.groups_[indexO]);
    }
    store_->mergeReaders(other.store());
    branchMapperPtr_->mergeMappers(other.branchMapperPtr());
  }

  EDProduct const*
  Principal::getIt(ProductID const& pid) const {
    assert(0);
    return 0;
  }

  void
  Principal::maybeFlushCache(TypeID const& tid, InputTag const& tag) const {
    if (tag.typeID() != tid ||
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
  Principal::swapBase(Principal& iOther) {
    std::swap(processHistoryPtr_, iOther.processHistoryPtr_);
    std::swap(processConfiguration_,iOther.processConfiguration_);
    std::swap(processHistoryModified_,iOther.processHistoryModified_);
    std::swap(groups_,iOther.groups_);
    std::swap(preg_, iOther.preg_);
    std::swap(branchMapperPtr_,iOther.branchMapperPtr_);
    std::swap(store_,iOther.store_);
  }
}
