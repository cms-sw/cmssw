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
    branchType_(bt)
  {
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
    if (0==g.get()) {
      return g;
    }
    if (resolveProv && (g->provenanceAvailable() || g->onDemand())) {
      if(g->onDemand()) {
         //must execute the unscheduled to get the provenance
         this->resolveProduct(*g, true);
         //check if this failed (say because of a caught exception)
         if( 0 == g->product()) {
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

    BasicHandleVec results;

    int nFound = findGroups(productType,
                            preg_->productLookup(),
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound) );
      *whyFailed
	<< "getBySelector: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
      return BasicHandle(whyFailed);
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getBySelector: Found "<<nFound<<" products rather than one which match all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    return results[0];
  }

  namespace  {
    class GetByLabelSelector : public SelectorBase {
    public:
      //NOTE: use const char* instead of string to avoid temporary creation of memory
      GetByLabelSelector(char const* iModule,
                         char const* iProductInstance,
                         char const* iProcessName):
      module_(iModule),
      productInstance_(iProductInstance),
      processName_(strlen(iProcessName)?iProcessName:static_cast<const char*> (0)) {}
      
      bool doMatch(ConstBranchDescription const& p) const {
        return ( (0 == strcmp(module_,p.moduleLabel().c_str()) ) &&
                 (0 == strcmp(productInstance_, p.productInstanceName().c_str())) &&
                 ( (0 == processName_) || (0 == strcmp(processName_, p.processName().c_str())) ) );
      }
      
      GetByLabelSelector* clone() const {
        return new GetByLabelSelector(*this);
      }
      
    private:
      char const * const module_;
      char const * const productInstance_;
      char const * const processName_;
  };
}
  
  BasicHandle
  Principal::getByLabel(TypeID const& productType,
			std::string const& label,
			std::string const& productInstanceName,
			std::string const& processName) const
  {

    BasicHandleVec results;
    
    GetByLabelSelector sel(label.c_str(),productInstanceName.c_str(),processName.c_str());
    
    int nFound = findGroups(productType,
                            preg_->productLookup(),
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound) );
      *whyFailed
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n"
	<< (processName.empty() ? "" : "Looking for process: ") << processName << "\n";
      return BasicHandle(whyFailed);
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found "<<nFound<<" products rather than one which match all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n"
	<< (processName.empty() ? "" : "Looking for process: ") << processName << "\n";
    }
    return results[0];
  }
 

  void 
  Principal::getMany(TypeID const& productType, 
		     SelectorBase const& sel,
		     BasicHandleVec& results) const {

    findGroups(productType,
               preg_->productLookup(),
               sel,
               results,
               false);

    return;
  }

  BasicHandle
  Principal::getByType(TypeID const& productType) const {

    BasicHandleVec results;

    edm::MatchAllSelector sel;

    int nFound = findGroups(productType,
                            preg_->productLookup(),
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound) );
      *whyFailed
	<< "getByType: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
      return BasicHandle(whyFailed);
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByType: Found "<<nFound <<" products rather than one which match all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    return results[0];
  }

  void 
  Principal::getManyByType(TypeID const& productType, 
			   BasicHandleVec& results) const {

    edm::MatchAllSelector sel;

    findGroups(productType,
               preg_->productLookup(),
               sel,
               results,
               false);
    return;
  }

  size_t
  Principal::getMatchingSequence(TypeID const& typeID,
				 SelectorBase const& selector,
				 BasicHandleVec& results,
				 bool stopIfProcessHasMatch) const {

    // One new argument is the element lookup container
    // Otherwise this just passes through the arguments to findGroups
    return findGroups(typeID,
                      preg_->elementLookup(),
                      selector,
                      results,
                      stopIfProcessHasMatch);
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
			TypeLookup const& typeLookup,
			SelectorBase const& selector,
			BasicHandleVec& results,
			bool stopIfProcessHasMatch) const {
    assert(results.empty());

    // A class without a dictionary cannot be in an Event/Lumi/Run.
    // First, we check if the class has a dictionary.  If it does not,
    // we return immediately.
    typedef TransientProductLookupMap TPLMap;
    const std::pair<TPLMap::const_iterator, TPLMap::const_iterator> range = typeLookup.equal_range(TypeInBranchType(typeID,branchType_));
    if(range.first == range.second) {
      return 0;
    }

    results.reserve(range.second - range.first);
    
    unsigned int processLevelFound = std::numeric_limits<unsigned int>::max();
    for(TPLMap::const_iterator it = range.first; it != range.second; ++it) {
      if( (it->processIndex() > processLevelFound) && stopIfProcessHasMatch ) {
        //this is for a later process and we've already found a match for an earlier process
        continue;
      }
        
      if( selector.match(*(it->branchDescription())) ) {
        
        //now see if the data is actual available
        SharedConstGroupPtr const& group = getGroupByIndex(it->index(), false, false, false);
        if(group.get() == 0) {
          continue;
        }
        // Skip product if not available.
        if (!group->productUnavailable()) {
          this->resolveProduct(*group, true);
	  // If the product is a dummy filler, group will now be marked unavailable.
          // Unscheduled execution can fail to produce the EDProduct so check
          if (!group->productUnavailable() && !group->onDemand()) {
            
            if(stopIfProcessHasMatch && it->processIndex() < processLevelFound) {
              results.clear();
              processLevelFound = it->processIndex();
            }
            
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
