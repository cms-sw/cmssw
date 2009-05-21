/**----------------------------------------------------------------------
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
                       BranchType bt,
		       ProcessHistoryID const& hist,
		       boost::shared_ptr<BranchMapper> mapper,
		       boost::shared_ptr<DelayedReader> rtrv) :
    EDProductGetter(),
    processHistoryPtr_(boost::shared_ptr<ProcessHistory>(new ProcessHistory)),
    processConfiguration_(&pc),
    processHistoryModified_(false),
    groups_(reg->constProductList().size(), SharedGroupPtr()),
    size_(0),
    preg_(reg),
    branchMapperPtr_(mapper),
    store_(rtrv),
    branchType_(bt) {
    if (hist.isValid()) {
      ProcessHistoryRegistry& history(*ProcessHistoryRegistry::instance());
      assert(history.notEmpty());
      bool found = history.getMapped(hist, *processHistoryPtr_);
      assert(found);
    }
    reg->productLookup().reorderIfNecessary(bt,*processHistoryPtr_,pc.processName());
    reg->elementLookup().reorderIfNecessary(bt,*processHistoryPtr_,pc.processName());
  }

  Principal::~Principal() {
  }

  Group*
  Principal::getExistingGroup(Group const& group) {
    ProductTransientIndex index = preg_->indexFrom(group.productDescription().branchID());
    if(index==ProductRegistry::kInvalidIndex) {
       return 0;
    }
    SharedGroupPtr ptr = groups_[index];
    assert(0==ptr.get() || BranchKey(group.productDescription())==BranchKey(ptr->productDescription()));
    return ptr.get();
  }

  void 
  Principal::addGroup_(std::auto_ptr<Group> group) {
    ConstBranchDescription const& bd = group->productDescription();
    assert (!bd.className().empty());
    assert (!bd.friendlyClassName().empty());
    assert (!bd.moduleLabel().empty());
    assert (!bd.processName().empty());
    SharedGroupPtr g(group);
    
    ProductTransientIndex index = preg_->indexFrom(bd.branchID());
    assert(index!= ProductRegistry::kInvalidIndex);
    groups_[index]=g;
    if(bool(g)){
      ++size_;
    }
  }

  void 
  Principal::replaceGroup(std::auto_ptr<Group> group) {
    ConstBranchDescription const& bd = group->productDescription();
    assert (!bd.className().empty());
    assert (!bd.friendlyClassName().empty());
    assert (!bd.moduleLabel().empty());
    assert (!bd.processName().empty());
    SharedGroupPtr g(group);
    ProductTransientIndex index = preg_->indexFrom(bd.branchID());
    assert(index!=ProductRegistry::kInvalidIndex);
    groups_[index]->replace(*g);
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
    if(index==ProductRegistry::kInvalidIndex){
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
            //behavior is the same as if the group wasn't there
            return SharedConstGroupPtr();
         }
      }
      this->resolveProvenance(*g);
    }
    if (resolveProd && !g->productUnavailable()) {
      this->resolveProduct(*g, fillOnDemand);
      if(g->onDemand() && 0 == g->product()) {
         //behavior is the same as if the group wasn't there
         return SharedConstGroupPtr();
      }
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
			std::string const& processName) const {

    BasicHandle result;
    
    bool found = findGroupByLabel(productType,
                                  preg_->productLookup(),
                                  label,
                                  productInstanceName,
                                  processName,
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

  void
  Principal::readImmediate() const {
    readProvenanceImmediate();
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      if (!(*i)->productUnavailable()) {
        resolveProduct(*(*i), false);
      }
    }
  }

  void
  Principal::readProvenanceImmediate() const {
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      if ((*i)->provenanceAvailable()) {
	resolveProvenance(**i);
      }
    }
    branchMapperPtr_->setDelayedRead(false);
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
			      BasicHandle& result) const {
    assert(!result.isValid());

    typedef TransientProductLookupMap TypeLookup;
    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not,
    // we return immediately.
    std::pair<TypeLookup::const_iterator, TypeLookup::const_iterator> const range = typeLookup.equal_range(TypeInBranchType(typeID,branchType_), moduleLabel, productInstanceName);
    if(range.first == range.second) {
      return false;
    }

    for(TypeLookup::const_iterator it = range.first; it != range.second; ++it) {
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

  void
  Principal::resolveProduct(Group const& g, bool fillOnDemand) const {
    if (g.productUnavailable()) {
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance() << '\n';
    }

    if (g.product()) return; // nothing to do.

    // Try unscheduled production.
    if (g.onDemand()) {
      if (fillOnDemand) unscheduledFill(g.productDescription().moduleLabel());
      return;
    }

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.productDescription());
    std::auto_ptr<EDProduct> edp(store_->getProduct(bk, this));

    // Now fix up the Group
    g.setProduct(edp);
  }

  void
  Principal::resolveProvenance(Group const& g) const {
    g.provenance()->setStore(branchMapperPtr_);
    g.provenance()->resolve();
  }

  OutputHandle
  Principal::getForOutput(BranchID const& bid, bool getProd) const {
    SharedConstGroupPtr const& g = getGroup(bid, getProd, true, false);
    if (g.get() == 0) {
      return OutputHandle();
    }
    if (getProd && (g->product() == 0 || !g->product()->isPresent()) &&
	    g->productDescription().present() &&
	    g->productDescription().branchType() == InEvent &&
            productstatus::present(g->productProvenancePtr()->productStatus())) {
        throw edm::Exception(edm::errors::LogicError, "Principal::getForOutput\n")
         << "A product with a status of 'present' is not actually present.\n"
         << "The branch name is " << g->productDescription().branchName() << "\n"
         << "Contact a framework developer.\n";
    }
    if (!g->product() && !g->productProvenancePtr()) {
      return OutputHandle();
    }
    return OutputHandle(g->product().get(), &g->productDescription(), g->productProvenancePtr());
  }

  Provenance
  Principal::getProvenance(BranchID const& bid) const {
    SharedConstGroupPtr const& g = getGroup(bid, false, true, true);
    if (g.get() == 0) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given branch id: "<< bid << "\n";
    }

    if (g->onDemand()) {
      unscheduledFill(g->productDescription().moduleLabel());
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
  Principal::swapBase(Principal& iOther) {
    std::swap(processHistoryPtr_, iOther.processHistoryPtr_);
    std::swap(processConfiguration_,iOther.processConfiguration_);
    std::swap(processHistoryModified_,iOther.processHistoryModified_);
    std::swap(groups_,iOther.groups_);
    std::swap(size_, iOther.size_);
    std::swap(preg_, iOther.preg_);
    std::swap(branchMapperPtr_,iOther.branchMapperPtr_);
    std::swap(store_,iOther.store_);
  }
}
